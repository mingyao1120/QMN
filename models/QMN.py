import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import DualTransformer
import math

from models.util import grad_mul_const
from models.loss import cal_nll_loss

# my_components
from models.modules.proposal import TAN2dProposal, Conv2dRanker
from models.modules.visual_mm_proposal import match_and_fuse_params
from models.modules.Interaction import InteractionEncoder

from models.modules.alignment_loss import VTCLoss

class QMN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.sigma = config["sigma"]
        self.use_negative = config['use_negative']
        self.num_props = config['num_props']
        self.max_epoch = config['max_epoch']
        self.gamma = config['gamma']
        
 
        self.frame_fc = nn.Linear(config['frames_input_size'], config['hidden_size']) # config['frames_input_size']
        self.word_fc = nn.Linear(config['words_input_size'], config['hidden_size'])
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.pred_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)

        self.trans = DualTransformer(**config['DualTransformer'])
        self.fc_comp = nn.Linear(config['hidden_size'], self.vocab_size)
        self.fc_gauss = nn.Linear(config['hidden_size'], self.num_props*2)
 
        self.word_pos_encoder = SinusoidalPositionalEmbedding(config['hidden_size'], 0, 20)

        self.max_video_len = 200

        # counterfactual
        self.fusion_mode = 'rubi'
        self.end_classif = True
        self.cross = False
        self.rectify = True
        self.weight = False
        self.gaussian_label = True
        self.beta_op, self.beta_on, self.beta_pn = 0.05, 0.15, 0.15
        # Q->A branch
        self.q_pos = copy.deepcopy(self.fc_comp)
        self.q_ref = copy.deepcopy(self.fc_comp)
        self.q_neg_1 = copy.deepcopy(self.fc_comp)
        self.q_neg_2 = copy.deepcopy(self.fc_comp)
        if self.end_classif:
            self.q_pos_additional = nn.Linear(self.vocab_size, self.vocab_size)
            self.q_ref_additional = nn.Linear(self.vocab_size, self.vocab_size)
            self.q_neg_1_additional = nn.Linear(self.vocab_size, self.vocab_size)
            self.q_neg_2_additional = nn.Linear(self.vocab_size, self.vocab_size)

        self.constant = nn.Parameter(torch.tensor(0.0))
        
        # my_proposal
        self.TDmaps = TAN2dProposal(downscale = 4)
        self.ranker = Conv2dRanker(dim = 256)
        self.ratio = config['ratio'] # 表明提案融合的程度
        self.margin_dis = config['margin_dis'] # 表明高斯分数更新的程度

        # self.fuser = VSLFuser()
        self.fuser = InteractionEncoder(config['hidden_size'], config['hidden_size'])
        self.vtcloss = VTCLoss()

        self.clip_len = 4
        self.conv1d_agg = nn.Conv1d(config['hidden_size'], config['hidden_size'], self.clip_len, self.clip_len)
        self.conv_dropout = nn.Dropout(0.2)
        

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, mode='train', **kwargs):
        bsz, n_frames, hiddent_dim = frames_feat.shape

        pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len)

        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        
        frames_feat_fused , A= self.fuser(frames_feat, words_feat, frames_mask, words_mask)
        # generate Gaussian masks
        enc_out, h, enc_out_v = self.trans(frames_feat + frames_feat_fused, frames_mask, words_feat + words_pos, words_mask, decoding=1)

        # global_word
        words_feat_global = enc_out[:, 0] # (32, 256)
        frames_feat_global = enc_out_v[:, -1]
        vtc_loss = self.vtcloss(frames_feat_global, words_feat_global)

        gauss_param = torch.sigmoid(self.fc_gauss(h[:, -1])).view(bsz,self.num_props, 2)
        
        pfeats, proposals, pmask = self.TDmaps(h[:, :-1], frames_mask[:, :-1].float())  # (32, 2500, 256)
        cands = self.ranker(pfeats * enc_out[:, 0].unsqueeze(1), pmask) # (32, 2500)
        
        my_props = (self.ranker.topk_confident(proposals, cands, pmask, k = 8) / self.max_video_len ) # 因为这里返回的是[0, 200]之间的值，使用sigmoid永远不能取到0.5以下的值
        my_props_i = my_props
        my_props[..., 0] = (my_props_i[..., 0] + my_props_i[..., 1]) / 2
        my_props[..., 1] = my_props_i[..., 1] - my_props_i[..., 0]

        gauss_param = match_and_fuse_params(gauss_param, my_props, self.ratio).view(bsz*self.num_props, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]

        
        props_len = n_frames//self.clip_len
        keep_idx = torch.linspace(0, n_frames-1, steps=props_len).long()
        frames_feat_sample = frames_feat[:, keep_idx]
        frames_mask = frames_mask[:, keep_idx]

        frames_feat_con = self.conv1d_agg(frames_feat[:, :-1].permute(0, 2, 1)).permute(0, 2, 1) # 使用卷积来聚合局部特征
        frames_feat = frames_feat_sample + self.conv_dropout(frames_feat_con)
        
        props_feat = frames_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, props_len, -1)  # torch.Size([256, 50, 256]) , 将采样后的特征复制了8份
        props_mask = frames_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)

        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width) # torch.Size([256, 50])
       
        global_word = enc_out[:, 0].unsqueeze(1).expand(bsz, self.num_props, props_feat.size(-1)).reshape(bsz*self.num_props, props_feat.size(-1)) # (32, 256) -> (256, 256)
        sim = torch.sigmoid(torch.bmm(props_feat, global_word.unsqueeze(-1))).squeeze()
        # 缩放到一定区间
        sim = sim * (2 * self.margin_dis) + (1 - self.margin_dis)
        
        gauss_weight = sim * gauss_weight
       
        words_feat, masked_words = self._mask_words(words_feat, words_len, weights=weights) 
        words_feat = words_feat + words_pos
        words_feat = words_feat[:, :-1]
        words_mask = words_mask[:, :-1]

        words_mask1 = words_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_id1 = words_id.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_feat1 = words_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, words_mask1.size(1), -1)

        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h, attn_weight,_ = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=pos_weight, need_weight=True)
        words_logit = self.fc_comp(h) # 字典长度分类的结果
        hq1, hq = words_feat1, words_feat
        # 重构损失
        q_out = self.counterfactual_module(words_logit, hq1, self.q_pos, self.q_pos_additional, mode='train')
        cf_loss_pos = self.counterfactual_loss(q_out, words_id1, words_mask1)   # (B*p)
 

        

        if self.use_negative:


            neg_1_weight, neg_2_weight = self.negative_proposal_mining(props_len, gauss_center, gauss_width, kwargs['epoch'])
            
            _, neg_h_1,_  = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=neg_1_weight)
            neg_words_logit_1 = self.fc_comp(neg_h_1)
            q_neg_out_1 = self.counterfactual_module(neg_words_logit_1, hq1, self.q_neg_1, self.q_neg_1_additional, mode='train')
            cf_loss_neg_1 = self.counterfactual_loss(q_neg_out_1, words_id1, words_mask1)   # (B*p)
  
            _, neg_h_2,_  = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=neg_2_weight)
            neg_words_logit_2 = self.fc_comp(neg_h_2)
            q_neg_out_2 = self.counterfactual_module(neg_words_logit_2, hq1, self.q_neg_2, self.q_neg_2_additional, mode='train')
            cf_loss_neg_2 = self.counterfactual_loss(q_neg_out_2, words_id1, words_mask1)   # (B*p)

            # TODO 优化ref_h
            _, ref_h,_  = self.trans(frames_feat, frames_mask, words_feat, words_mask, decoding=2) # 都是采样后的结果
            ref_words_logit = self.fc_comp(ref_h)
            q_ref_out = self.counterfactual_module(ref_words_logit, hq, self.q_ref, self.q_ref_additional, mode='train')
            cf_loss_ref = self.counterfactual_loss(q_ref_out, words_id, words_mask)   # (B)


        else:
            neg_words_logit_1 = None
            neg_words_logit_2 = None
            ref_words_logit = None

        cf_loss = 0 #cf_loss_pos + cf_loss_ref + cf_loss_neg_1 + cf_loss_neg_2

        if mode == 'test':

            words_logit = q_out['logits_cfvqa']

        return {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'ref_words_logit': ref_words_logit,
            'words_logit': words_logit,
            'words_id': words_id,
            'words_mask': words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
            'cf_loss': cf_loss,
            'cf_neg_loss_1': cf_loss_neg_1,
            'cf_neg_loss_2': cf_loss_neg_2,
            'cf_ref_loss': cf_loss_ref,
            'cf_pos_loss': cf_loss_pos,
            'vtc_loss': vtc_loss,
        }

    def counterfactual_loss(self, prediction, gt, q_mask, sim=None):
        v_pre = prediction['logits_tar']
        # q_pre = prediction['logits_src']
        z_q = prediction['z_nde']
        z_qkv = prediction['logits_all']
        # KL loss
        p_te = torch.nn.functional.softmax(z_qkv, -1).clone().detach()
        p_nde = torch.nn.functional.softmax(z_q, -1)
        kl_loss = - p_te * p_nde.log()
        kl_loss = kl_loss.sum(1).mean()
        # prediction loss
        loss_v, _ = cal_nll_loss(v_pre, gt, q_mask)
        loss_z, _ = cal_nll_loss(z_qkv, gt, q_mask)
        cf_loss = kl_loss + loss_v + loss_z
        # cf_loss = torch.mean(kl_loss) + torch.mean(loss_v) + torch.mean(loss_z)
        return cf_loss

    def counterfactual_module(self, fusion, tar, tar_head, tar_add=None, mode='train'):
        out = {}

        # tar = grad_mul_const(tar, 0.0)  # don't backpropagate
        tar_pred = tar_head(tar).squeeze(-1)  # N * T * D -> N * T

        # src = grad_mul_const(src, 0.0)  # don't backpropagate
        # src_pred = src_head(src)  # N * D -> N * T

        # both q, k and v are the facts
        z_qkv = self.fusion(fusion=fusion, target=tar_pred,
                            fusion_fact=True, target_fact=True)  # te = total effect
        # q is the fact while k and v are the counterfactuals
        z_q = self.fusion(fusion=fusion, target=tar_pred,
                          fusion_fact=False, target_fact=True)  # nie = natural indirect effect

        logits_cfvqa = z_qkv - z_q

        if self.end_classif:
            tar_out = tar_add(tar_pred)  # N * T
            # src_out = src_add(src_pred)  # N * T -> N * T
        else:
            tar_out = tar_pred
            # src_out = src_pred

        if mode == 'train':
            out['logits_all'] = z_qkv  # for optimization
            # out['logits_vq'] = logits  # predictions of the original VQ branch, i.e., NIE
            out['logits_cfvqa'] = logits_cfvqa  # predictions of CFVQA, i.e., TIE
            out['logits_tar'] = tar_out  # for optimization
            # out['logits_src'] = src_out  # for optimization
            out['z_nde'] = self.fusion(fusion.clone().detach(), tar_pred.clone().detach(),
                                       fusion_fact=False, target_fact=True)  # z_q for kl optimization with no grad
            return out
        else:
            return logits_cfvqa

    def fusion(self, fusion, target, fusion_fact=False, target_fact=False):

        fusion, target = self.transform(fusion, target, fusion_fact=fusion_fact,
                                                target_fact=target_fact)

        if self.fusion_mode == 'rubi':
            z = fusion * torch.sigmoid(target)

        elif self.fusion_mode == 'hm':
            z = fusion * target
            z = torch.log(z + eps) - torch.log1p(z)

        elif self.fusion_mode == 'sum':
            z = fusion + target
            z = torch.log(torch.sigmoid(z) + eps)

        return z

    def transform(self, fusion, target, fusion_fact=False, target_fact=False):
        gpu_id = self.constant.device
        # cuda_str = 'cuda:{}'.format(gpu_id)
        device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
        # torch.cuda.set_device(gpu_id)
        if not fusion_fact:
            fusion = self.constant * torch.ones_like(fusion).to(device)

        if not target_fact:
            target = self.constant * torch.ones_like(target).to(device)

        if self.fusion_mode == 'hm':
            fusion = torch.sigmoid(fusion)
            target = torch.sigmoid(target)

        return fusion, target
    
    def generate_gauss_weight(self, props_len, center, width):
        # pdb.set_trace()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        w = 0.3989422804014327
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))
        
        # ori_gau = weight / weight.max(dim=-1, keepdim=True)[0]
        # # Calculate the mean of the last dimension
        # mean_value = ori_gau.mean(dim=-1, keepdim=True)
        # # Threshold values: set values greater than the mean to 1, others remain unchanged
        # ori_gau_thresholded = torch.where(ori_gau > mean_value, torch.tensor(1.0).to(ori_gau.device), ori_gau)
        # # ori_gau_thresholded now contains the final result


        return weight/weight.max(dim=-1, keepdim=True)[0]

    def negative_proposal_mining(self, props_len, center, width, epoch):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma/2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w/w1*torch.exp(-(pos-c)**2/(2*w1**2))
            return y1/y1.max(dim=-1, keepdim=True)[0]

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center-width/2, min=0)
        left_center = left_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5
        right_width = torch.clamp(1-center-width/2, min=0)
        right_center = 1 - right_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5

        left_neg_weight = Gauss(weight, left_center, left_center)
        right_neg_weight = Gauss(weight, 1-right_center, right_center)

        return left_neg_weight, right_neg_weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1) 
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            if l < 1:
                continue
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

