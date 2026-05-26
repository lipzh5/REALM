"""Features extracted by ourselves, but following the rules of dataset organization used in ViCo Challenge"""
from bisect import bisect_right
from math import ceil
import numpy as np
import os
import os.path as osp
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist
from copy import deepcopy
from box import Box
import torch.nn.utils.rnn as rnn_utils
from decord import VideoReader, cpu
import random
import pickle
import os
import os.path as osp
from scipy.io import loadmat

CUR_DIR = osp.dirname(osp.abspath(__file__))



SENTIMENT2LABEL = {
    'neutral': 0,
    'positive': 1,
    'negative': 2,
}

def split_coeff(coeffs): # for deep3d reconstruction
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }

def get_target_coeffs(mat_data):
    coeffs_full = mat_data['coeff'] # (T, 257)
    coeff_dict = split_coeff(coeffs_full)
    target_coeff = np.concatenate([coeff_dict['exp'], coeff_dict['angle'], coeff_dict['trans']], axis=1)
    return target_coeff


def get_mean_std(data_root, who):
    stat_path = osp.join(data_root, f'{who}_face_70d_stat.npz')
    stats = np.load(stat_path)
    mean_face = stats['mean']
    std_face = stats['std']
    return mean_face.astype(np.float32), std_face.astype(np.float32)



class VicoDataset(data.Dataset):
    def __init__(self, config, split, clip_length=150, stride=4):
        self.split = split
        self.stride = stride
        self._root_dir = osp.join(CUR_DIR, '../../data/vico')
        self.audio_dir = osp.join(self._root_dir,'audio-features')
        self.mfcc_dir = osp.join(self._root_dir, 'mfcc-features')
        self.audio_dual_stream = getattr(config.data, 'audio_dual_stream', False)

        if config.model.audio_dim == 1024:
            self.audio_dir = osp.join(self._root_dir, 'audio-features-emotion')
        self.video_dir = osp.join(CUR_DIR, '../../data/vico_raw/videos')
        coeffs_dir = osp.join(CUR_DIR, '../../data/vico_raw/deep3d_coeffs/cropped')
    
        mean_face, std_face = get_mean_std(self._root_dir, who='listener')
        self.transform_3dmm = transforms.Lambda(lambda e: (e - mean_face) / std_face)


        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])
        meta_path = osp.join(self._root_dir, 'RLD_data.csv')
        meta_data = pd.read_csv(meta_path)
        raw_data = []

        pad_test_to = getattr(config.data, 'pad_test_to', 0)
        for i, row in meta_data.iterrows():
            if row.data_split != self.split:
                continue
            audio_feat = np.load(osp.join(self.audio_dir, f'{row.audio}.npy'))
            # print(f'audio feat: {audio_feat.shape}, mfcc feat: {mfcc_feat.shape}')
            
            mat_data = loadmat(osp.join(coeffs_dir, f'{row.speaker}.mat'))
            coeffs_speaker = (get_target_coeffs(mat_data))

            mat_data = loadmat(osp.join(coeffs_dir, f'{row.listener}.mat'))
            coeffs_listener = (get_target_coeffs(mat_data))

            nframes = min(len(audio_feat), len(coeffs_speaker), len(coeffs_listener))

            if self.audio_dual_stream:
                mfcc_feat = np.load(osp.join(self.mfcc_dir, f'{row.audio}.npy'))
                concat_audio_feat = np.concatenate([audio_feat[:nframes], mfcc_feat[:nframes]], axis=1)
            else:
                concat_audio_feat = audio_feat[:nframes]
            data_entry = {
                'audio': concat_audio_feat,
                'speaker': coeffs_speaker[:nframes],
                'listener': coeffs_listener[:nframes],
            }

            if split == 'test' or split == 'ood':
                frame_indexs = list(range(nframes))
                if 0 < nframes < pad_test_to:
                    print(f'!!! Pad Test Length to {pad_test_to}')
                    # Repeat the last frame index: frame_indexs[-1]
                    padding = [frame_indexs[-1]] * (32 - nframes)
                    frame_indexs.extend(padding)
                data_entry['frame_indexs'] = frame_indexs
                raw_data.append(data_entry)
            else:
                if nframes > clip_length:
                    for i in range(0, nframes - clip_length + 1, self.stride):
                        entry = {
                            'audio': data_entry['audio'],
                            'speaker': data_entry['speaker'],
                            'listener': data_entry['listener'],
                            'frame_indexs': list(range(i, i+ clip_length)),}
                        raw_data.append(entry)
                else:
                    data_entry['frame_indexs'] = list(range(nframes)),
                    raw_data.append(data_entry)

        self.data = raw_data
        self.audio_zeros = getattr(config.data, 'audio_zeros', False)
        self.motion_zeros = getattr(config.data, 'motion_zeros', False)

    def __getitem__(self, idx):
        frame_indexs = self.data[idx]['frame_indexs']
        audio = torch.from_numpy(self.data[idx]['audio'])[frame_indexs]
        coeff_listener = torch.from_numpy(self.data[idx]['listener'])[frame_indexs]
        coeff_speaker = torch.from_numpy(self.data[idx]['speaker'])[frame_indexs]

        speaker_coeff = self.transform_3dmm(coeff_speaker)
        listener_coeff = self.transform_3dmm(coeff_listener)

        if self.audio_zeros:
            audio = torch.zeros_like(audio)
        if self.motion_zeros:
            speaker_coeff = torch.zeros_like(speaker_coeff)
      
        return audio, speaker_coeff, listener_coeff

    def __len__(self):
        return len(self.data)

def get_dataset(config, split):
    clip_length = config.data.clip_length
    # load_sentiment = config.data.load_sentimen
    dataset = VicoDataset(
        config,
        split,
        clip_length,
    )
    print(f'len dataset: {len(dataset)}')
    return dataset

def collate_fn(batch):
    audio = [e[0] for e in batch]
    speaker_coeffs = [e[1] for e in batch]
    listener_coeffs = [e[2] for e in batch]
    lengths = torch.from_numpy(np.array([e.size(0) for e in speaker_coeffs]))

    audio = rnn_utils.pad_sequence(audio, batch_first=True)
    speaker_coeffs = rnn_utils.pad_sequence(speaker_coeffs, batch_first=True)
    listener_coeffs = rnn_utils.pad_sequence(listener_coeffs, batch_first=True)

    res_dict = {
            'audio': audio,
            'speaker_face': speaker_coeffs,
            'listener_face': listener_coeffs,
            'lengths': lengths
    }

    return res_dict


def get_data_loader(config, split):
    dataset = get_dataset(config, split)
    
    loader = DataLoader(
        dataset=dataset,
        shuffle=True if split=='train' else False,
        # sampler=sampler,
        collate_fn=collate_fn,
        batch_size=config.train.batch_size,
        drop_last=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )
    return loader
