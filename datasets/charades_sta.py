import h5py
import numpy as np

from datasets.base import BaseDataset, build_collate_data


class CharadesSTA(BaseDataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        super().__init__(data_path, vocab, args, **kwargs)
        self.collate_fn = build_collate_data(args['max_num_frames'], args['max_num_words'],
                                             args['frame_dim'], args['word_dim'])

    def _load_frame_features(self, vid):
        if not isinstance(self.args['feature_path'],list):
            with h5py.File(self.args['feature_path'], 'r') as fr:
                return np.asarray(fr[vid]).astype(np.float32)
        else:
            with h5py.File(self.args['feature_path'][0], 'r') as fr_0:
                with h5py.File(self.args['feature_path'][1], 'r') as fr_1:
                    clip_f = np.asarray(fr_0[vid]).astype(np.float32)
                    sf_f = np.asarray(fr_1[vid]).astype(np.float32)
                    # print()
                    if clip_f.shape != sf_f.shape: # 若二者长度不等，则取小值
                        sf_f = sf_f[:clip_f.shape[0]]
                    return np.concatenate(( sf_f, clip_f),axis=-1) # 特征维度拼接

    def collate_data(self, samples):
        return self.collate_fn(samples)
