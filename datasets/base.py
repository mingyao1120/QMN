import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_json
import nltk

# 数据增强函数
def shift_timestamp_and_feature(timestamp, feature): 
    # 输入的时间戳要是归一化后的结果
    # 随机生成一个平移距离，可以根据需要设置范围和概率分布 
    shift = torch.rand(1) * (timestamp[1] - timestamp[0])  # 这里可以根据相关时刻的长度来定义，长的移动范围就大，短的移动范围就小
    # 将时间戳加上平移距离，注意要保证时间戳在0到1之间 
    # 需要限制一下，不然会丢失一些数据
    if timestamp[0] == 0:
        shift = torch.abs(shift)
    if timestamp[1] == 1:
        shift = -torch.abs(shift)
    timestamp = torch.clamp(timestamp + shift, 0, 1) 
    
    # 根据平移距离和视频特征的长度计算平移的帧数 
    shift_frame = int(shift * len(feature)) 
    # 如果平移距离为正，则将视频特征向左平移，并用0填充右边空缺的部分 
    feature = torch.cat([feature[- shift_frame:], feature[:- shift_frame]], dim=0) 
    # if shift_frame > 0: 
    #     #  torch.cat([feature[- shift_frame:], feature[:- shift_frame]], dim=0) 
    #     feature = torch.cat([feature[shift_frame:], feature[:shift_frame]], dim=0)  # torch.zeros(shift_frame, feature.size(1))
    #     # 如果平移距离为负，则将视频特征向右平移，并用0填充左边空缺的部分 
    # elif shift_frame < 0: 
    #     # torch.cat([feature[- shift_frame:] ,feature[:- shift_frame],], dim=0)
    #     feature = torch.cat([feature[:shift_frame], feature[shift_frame:]], dim=0)  # torch.zeros(-shift_frame, feature.size(1))
    #     # 返回移动后的时间戳和视频特征 
    return timestamp, feature

class BaseDataset(Dataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        self.vocab = vocab
        self.args = args
        self.data = load_json(data_path) # 先读入对应的json文件
        self.ori_data = self.data
        self.max_num_frames = args['max_num_frames']
        self.max_num_words = args['max_num_words']
        self.IsTrainmode = 'train' in data_path

        self.keep_vocab = dict()
        for w, _ in vocab['counter'].most_common(args['vocab_size']):
            self.keep_vocab[w] = self.vocab_size

    def _load_frame_features(self, vid):
        raise NotImplementedError

    def _sample_frame_features(self, frames_feat):
        num_clips = self.num_clips
        keep_idx = np.arange(0, num_clips + 1) / num_clips * len(frames_feat)   # 不论长短都是均匀取200（max_num_frame）帧
        keep_idx = np.round(keep_idx).astype(np.int64)
        keep_idx[keep_idx >= len(frames_feat)] = len(frames_feat) - 1
        frames_feat1 = []
        for j in range(num_clips):
            s, e = keep_idx[j], keep_idx[j + 1]
            assert s <= e
            if s == e:
                frames_feat1.append(frames_feat[s])
            else:
                frames_feat1.append(frames_feat[s:e].mean(axis=0))
        return np.stack(frames_feat1, 0)

    @property
    def num_clips(self):
        return self.max_num_frames

    @property
    def vocab_size(self):
        return len(self.keep_vocab) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid, duration, timestamps, sentence = self.data[index] # 取出json文件中的一行 # 每行的query应该都不同
        duration = float(duration)
        # idea0这里能不能进行一些数据增强呢？

        weights = [] # Probabilities to be masked，读入数据就对文本进行随机掩码？
        words = []
        for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
            word = word.lower()
            if word in self.keep_vocab: # 需要保留（不能被掩码）的单词
                if 'NN' in tag:
                    weights.append(2)
                elif 'VB' in tag:
                    weights.append(2)
                elif 'JJ' in tag or 'RB' in tag:
                    weights.append(2)
                else:
                    weights.append(1)
                words.append(word)
        words_id = [self.keep_vocab[w] for w in words]

        # Glove+word2vec
        words_feat = [self.vocab['id2vec'][self.vocab['w2id'][words[0]]].astype(np.float32)] # placeholder for the start token，留出首个位置给开始token
        words_feat.extend([self.vocab['id2vec'][self.vocab['w2id'][w]].astype(np.float32) for w in words])
        # print(len(words_feat))
        frames_feat = self._sample_frame_features(self._load_frame_features(vid)) # 这里一旦读取就是整个视频都读取进来呀
        
        # 增强的思路，随机将时间戳进行左右移动，然后读取出来的整段视频对应的视频特征token也要进行移动
        # if self.IsTrainmode and torch.rand(1) < 0.2: # 以20%的几率来反转
        #     # print('换！')
        #     timestamps = torch.Tensor(timestamps) / duration
        #     Shifted_timestamps, Shifted_feats = shift_timestamp_and_feature(timestamps , torch.Tensor(self._load_frame_features(vid)))
        #     # print(timestamps, Shifted_timestamps)
        #     frames_feat = self._sample_frame_features(Shifted_feats)
        #     timestamps  = Shifted_timestamps * duration
        # else:
        #     # print('不换！')
        #     frames_feat = self._sample_frame_features(self._load_frame_features(vid))
            
        
        return {
            'frames_feat': frames_feat,
            'words_feat': words_feat,
            'words_id': words_id,
            'weights': weights,
            'raw': [vid, duration, timestamps, sentence]
        }

# def start_end_collate(batch):
#     batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

#     model_inputs_keys = batch[0]["model_inputs"].keys()
#     batched_data = dict()
#     for k in model_inputs_keys:
#         if k == "span_labels":
#             batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
#             continue
#         if k in ["saliency_pos_labels", "saliency_neg_labels"]:
#             batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
#             continue
#         if k == "saliency_all_labels":
#             pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
#             # print(pad_data, mask_data)
#             batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
#             continue

#         batched_data[k] = pad_sequences_1d(
#             [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
#     return batch_meta, batched_data


def build_collate_data(max_num_frames, max_num_words, frame_dim, word_dim):
    def collate_data(samples):
        bsz = len(samples)
        batch = {
            'raw': [sample['raw'] for sample in samples],
        }

        frames_len = []
        words_len = []

        for i, sample in enumerate(samples):
            frames_len.append(min(len(sample['frames_feat']), max_num_frames))
            words_len.append(min(len(sample['words_id']), max_num_words))

        frames_feat = np.zeros([bsz, max_num_frames, frame_dim]).astype(np.float32)
        words_feat = np.zeros([bsz, max(words_len) + 1, word_dim]).astype(np.float32)
        words_id = np.zeros([bsz, max(words_len)]).astype(np.int64)
        weights = np.zeros([bsz, max(words_len)]).astype(np.float32)
        for i, sample in enumerate(samples):
            frames_feat[i, :len(sample['frames_feat'])] = sample['frames_feat']
            keep = min(len(sample['words_feat']), words_feat.shape[1])
            words_feat[i, :keep] = sample['words_feat'][:keep]
            keep = min(len(sample['words_id']), words_id.shape[1])
            words_id[i, :keep] = sample['words_id'][:keep]
            keep = min(len(sample['weights']), weights.shape[1])
            tmp = np.exp(sample['weights'][:keep])
            weights[i, :keep] = tmp / np.sum(tmp)

        batch.update({
            'net_input': {
                'frames_feat': torch.from_numpy(frames_feat),
                'frames_len': torch.from_numpy(np.asarray(frames_len)),
                'words_feat': torch.from_numpy(words_feat),
                'words_id': torch.from_numpy(words_id),
                'weights': torch.from_numpy(weights),
                'words_len': torch.from_numpy(np.asarray(words_len)),
            }
        })
        return batch

    return collate_data
