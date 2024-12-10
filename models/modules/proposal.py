from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


def mask_pooling(ntensor, nmask, mode='avg', dim=-1):
    """mark pooling over the given tensor
    """
    if mode in ['avg', 'sum']:
        ntensor = (ntensor * nmask).sum(dim)
        if mode == 'avg':
            ntensor /= nmask.sum(dim).clamp(min=1e-10)
    elif mode == 'max':
        ntensor = mask_logits(ntensor, nmask)
        ntensor = ntensor.max(dim)[0]
    else:
        raise NotImplementedError('Only sum/average/max pooling have been implemented')
    return ntensor


def downscale1d(x, scale=1, dim=None, mode='max'):
    # pad at tail
    padding = (scale - x.shape[-1] % scale) % scale
    x = F.pad(x, (0, padding), value=(0 if mode=='sum' else -1e-30))
    # downscale
    if mode == 'sum':
        kernel = x.new_ones(1, 1, scale)
        return F.conv1d(x, kernel, stride=scale, padding=0)
    elif mode == 'max':
        return F.max_pool1d(x, scale, stride=scale, padding=0)
    raise NotImplementedError('downscale1d implemented only "max" and "sum" mode')


class TAN2dProposal(nn.Module):

    def __init__(self, downscale=8, windows=[16], **kwargs):
        super().__init__()
        # downscale before generating proposals
        self.scale = downscale
        # pooling layers
        self.windows = windows
        layers = []
        for i, window in enumerate(windows):
            layers.extend([nn.MaxPool1d(1, 1) if i == 0 else nn.MaxPool1d(3, 2)]) # AvgPool1d MaxPool1d
            layers.extend([nn.MaxPool1d(2, 1) for _ in range(window - 1)])
        self.layers = nn.ModuleList(layers)

    def forward(self, feats=None, mask=None, **kwargs):
        assert None not in [feats, mask]
        # set all invalid features to a very small values
        # to avoid their impact in maxpooling
        feats = mask_logits(feats, mask.unsqueeze(-1))
        # apply downscale first
        feats = downscale1d(feats.transpose(1, 2), scale=self.scale, mode='max')  # 'avg', 'sum' 'max'
        scaled_mask = downscale1d(mask.unsqueeze(1), scale=self.scale, mode='max').squeeze(1)
        B, D, N = feats.shape
        mask2d = mask.new_zeros(B, N, N)
        feat2d = feats.new_zeros(B, D, N, N)
        offset, stride = -1, 1
        for i, window in enumerate(self.windows):
            for j in range(window):
                layer = self.layers[i * len(self.windows) + j]
                if feats.shape[-1] < layer.kernel_size: break
                offset += stride
                start, end = range(0, N - offset, stride), range(offset, N, stride)
                # assume valid features are continual
                mask2d[:, start, end] = scaled_mask[:, end]
                feats = layer(feats)
                feat2d[:,:,start,end] = feats
            stride *= 2
        # mask invalid proposal features to 0
        feat2d *= mask2d.unsqueeze(1)
        # (B, D, N, N) -> (B, N, N, D)
        feat2d = feat2d.permute(0, 2, 3, 1)
        # generate boundary
        bounds = torch.arange(0, N, device=feats.device)
        bounds = bounds.view(1, -1).repeat(B, 1)
        # (B, N, N, 2)
        bounds = torch.stack([bounds.unsqueeze(-1).repeat(1, 1, N) * self.scale,
            (bounds.unsqueeze(1).repeat(1, N, 1) + 1) * self.scale - 1], dim=-1)
        # set the largest boundary to the number of items in each sample
        bounds = torch.min(bounds, mask.sum(-1).view(-1, 1, 1, 1).long() - 1)
        # mask invalid proposal
        bounds *= mask2d.unsqueeze(-1).long()
        # make sure for all the valid proposals
        # its endpoint should be greater or equal to its start points
        assert ((bounds[...,1] >= bounds[...,0]) + ~mask2d.bool()).all()
        # flatten all proposals as output
        feat2d = feat2d.view(B, N * N, D)
        bounds = bounds.view(B, N * N, 2)
        mask2d = mask2d.view(B, N * N)
        return feat2d, bounds, mask2d


def calculate_batch_iou(i0, i1):
    assert i0.dim() == i1.dim()
    union = (torch.min(i0[...,0], i1[...,0]), torch.max(i0[...,1], i1[...,1]))
    inter = (torch.max(i0[...,0], i1[...,0]), torch.min(i0[...,1], i1[...,1]))
    iou = 1. * (inter[1] - inter[0]).clamp(min=0) / (union[1] - union[0]).clamp(min=1e-5)
    return iou

class Conv2dRanker(nn.Module):

    def __init__(self, dim=128, kernel_size=3, num_layers=4, **kwargs): # ks = 3
        super().__init__()
        self.kernel = kernel_size
        self.encoders = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2) 
        for _ in range(num_layers)])
        self.predictor = nn.Conv2d(dim, 1, 1)

    @staticmethod
    def get_padded_mask_and_weight(mask, conv):
        masked_weight = torch.round(F.conv2d(mask.clone().float(), 
            mask.new_ones(1, 1, *conv.kernel_size),
            stride=conv.stride, padding=conv.padding, dilation=conv.dilation))
        masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0] #conv.kernel_size[0] * conv.kernel_size[1]
        padded_mask = masked_weight > 0
        return padded_mask, masked_weight

    def forward(self, x, mask):
        # convert to 2d if input is flat
        if x.dim() < 4:
            B, N2, D = x.shape
            assert int(math.sqrt(N2)) == math.sqrt(N2)
            N = int(math.sqrt(N2))
            x2d, mask2d = x.view(B, N, N, D), mask.view(B, N, N)
        else:
            x2d, mask2d = x, mask
        # x: (<bsz>, <num>, <num>, <dim>) -> (<bsz>, <dim>, <num>, <num>)
        x2d, mask2d = x2d.permute(0, 3, 1, 2), mask2d.unsqueeze(1)
        for encoder in self.encoders:
            # mask invalid features to 0
            x2d = F.relu(encoder(x2d * mask2d))
            _, weights = self.get_padded_mask_and_weight(mask2d, encoder)
            x2d = x2d * weights
        # preds: (<bsz>, <num>, <num>)
        preds = self.predictor(x2d).view_as(mask)
        preds = mask_logits(preds, mask)
        return preds.sigmoid()

    @staticmethod
    def topk_confident(bounds, scores, mask, moments=None, threshold=0.5, k=1):
        if moments is not None:
            # compute the overlaps between proposals and ground-truth
            overlaps = calculate_batch_iou(bounds, moments.unsqueeze(1))
            # set the scores of proposals with 
            # insufficient overlaps with the ground-truth to -inf
            is_cand = (overlaps >= threshold) * mask
        else:
            is_cand = mask
        masked_scores = mask_logits(scores, is_cand)
        # get topk proposals
        cands_idx = masked_scores.topk(k, dim=1)[1]
        cands = bounds.gather(1, cands_idx.unsqueeze(-1).repeat(1, 1, 2))
        if moments is not None:
            # in training
            # use the ground-truth moment if there is no proposal whose overlaps 
            # with the ground-truth is greater than the threshold
            # for example, when the threshold equal to 1
            has_cand = (is_cand.sum(-1) > 0).view(-1, 1, 1)
            cands = cands * has_cand + moments.unsqueeze(1) * (~has_cand)
        return cands

    @staticmethod
    def compute_loss(moments, bounds, scores, mask, min_iou=0.3, max_iou=0.7):
        assert scores.shape == mask.shape and min_iou <= max_iou
        ious = calculate_batch_iou(bounds, moments.unsqueeze(1))
        if min_iou == max_iou:
            targets = (ious >= min_iou).float()
        else:
            targets = (ious - min_iou) / (max_iou - min_iou)
            targets = targets.clamp(min=0, max=1)
        return F.binary_cross_entropy(
            scores.masked_select(mask.bool()),
            targets.masked_select(mask.bool()))

