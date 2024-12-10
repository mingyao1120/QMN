import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# clip-text
class CT_Loss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CT_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, vid_feat, txt_feat, pos_mask, src_vid_mask=None, src_txt_mask=None):
        # vid_feat: (bs, t, d)
        # txt_feat: (bs, n, d)
        # pos_mask: (bs, t)
        # src_vid_mask: (bs, t) or None
        # src_txt_mask: (bs, n) or None
        bs = vid_feat.size(0)
        t = vid_feat.size(1)
        n = txt_feat.size(1)
        d = vid_feat.size(2)
        # normalize the feature vectors
        vid_feat = F.normalize(vid_feat, dim=2) # (bs, t, d)
        txt_feat = F.normalize(txt_feat, dim=2) # (bs, n, d)
        # compute the global text feature by mean pooling
        if src_txt_mask is not None:
            src_txt_mask = src_txt_mask.unsqueeze(-1) # (bs, n, 1)
            txt_feat = txt_feat * src_txt_mask # (bs, n, d)
            txt_global = torch.sum(txt_feat, dim=1) / torch.sum(src_txt_mask, dim=1) # (bs, d)
        else:
            txt_global = torch.mean(txt_feat, dim=1) # (bs, d)
        # compute the similarity matrix
        sim_mat = torch.bmm(vid_feat, txt_global.unsqueeze(-1)).squeeze(-1) # (bs, t)
        # apply the video mask if given
        if src_vid_mask is not None:
            sim_mat = sim_mat * src_vid_mask # (bs, t)
        # compute the logits and labels
        logits = sim_mat / self.temperature # (bs, t)
        labels = pos_mask.long() # (bs, t)
        # compute the binary cross entropy loss with logits
        loss = F.binary_cross_entropy_with_logits(logits, labels.float()) # scalar
        # return the loss
        return loss
    
# clip-clip
class CC_Loss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CC_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, vid_feat, pos_idx, src_vid_mask=None):
        norm = torch.norm(vid_feat, dim=-1) # (b, t)
        # 将norm扩展为(b, t, 1)和(b, 1, t)的形状
        norm1 = norm.unsqueeze(-1) # (b, t, 1)
        norm2 = norm.unsqueeze(1) # (b, 1, t)
        # 计算分母矩阵
        denom = torch.bmm(norm1, norm2) # (b, t, t)
        # 计算分子矩阵
        num = torch.bmm(vid_feat, vid_feat.transpose(1, 2)) # (b, t, t)
        # 计算余弦相似度矩阵
        sim = torch.div(num, denom) # (b, t, t)
        # 将pos_idx扩展为(b, t, 1)和(b, 1, t)的形状
        pos_idx1 = pos_idx.unsqueeze(-1) # (b, t, 1)
        pos_idx2 = pos_idx.unsqueeze(1) # (b, 1, t)
        # 计算正样本标签矩阵
        pos_label = torch.eq(pos_idx1, pos_idx2).float() # (b, t, t)

        loss_fn = nn.BCEWithLogitsLoss()
        # 计算损失值
        loss = loss_fn(sim, pos_label)
        return loss
    

class VTCLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(VTCLoss, self).__init__()
        self.temperature = temperature

    def forward(self, src_txt, src_vid):
        # src_txt: (bs, h_dim)
        # src_vid: (bs, h_dim)
        bs = src_txt.size(0)
        h_dim = src_txt.size(1)
        # normalize the feature vectors
        src_txt = F.normalize(src_txt, dim=1)
        src_vid = F.normalize(src_vid, dim=1)
        # compute the similarity matrix
        sim_mat = torch.mm(src_txt, src_vid.t()) # (bs, bs)
        # create the positive and negative masks
        pos_mask = torch.eye(bs).bool().to(sim_mat.device) # (bs, bs)
        neg_mask = ~pos_mask # (bs, bs)
        # compute the logits and labels
        logits = sim_mat / self.temperature # (bs, bs)
        labels = torch.arange(bs).to(sim_mat.device) # (bs,)
        # compute the cross entropy loss for text-to-video and video-to-text
        loss_t2v = F.cross_entropy(logits, labels) # scalar
        loss_v2t = F.cross_entropy(logits.t(), labels) # scalar
        # return the average loss
        return (loss_t2v + loss_v2t) / 2