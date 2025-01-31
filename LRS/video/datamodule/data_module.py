import os

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule

from datamodule.av_dataset import AVDataset
from .transforms import VideoTransform, AudioTransform
from glob import glob

def pad(samples, pad_val=0.0):
    """ https://github.com/facebookresearch/av_hubert/blob/593d0ae8462be128faab6d866a3a926e2955bde1/avhubert/hubert_dataset.py#L517 """
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])

    # 디버깅: 크기 출력
    print(f"Samples lengths: {lengths}, Max size: {max_size}, Sample shape: {sample_shape})") # 모노 1채널로 잘 바꿨다면 오디오 샘플 쉐입은 빈 리스트인게 맞!
 
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths
# def pad(samples, pad_val=0.0):
#     lengths = [len(s) for s in samples]
#     max_size = max(lengths)
#     sample_shape = list(samples[0].shape[1:])

#     # Debugging
#     print(f"Samples lengths: {lengths}, Max size: {max_size}, Sample shape: {sample_shape}")

#     collated_batch = samples[0].new_full([len(samples), max_size] + sample_shape, pad_val)
#     for i, sample in enumerate(samples):
#         if len(sample) < max_size:
#             # Pad shorter samples
#             collated_batch[i, :len(sample)] = sample
#         elif len(sample) > max_size:
#             # Trim longer samples
#             collated_batch[i] = sample[:max_size]
#     return collated_batch, lengths

# def collate_pad(batch):
#     batch_out = {}
#     for data_type in batch[0].keys():
#         pad_val = -1 if data_type == "target" else 0.0
#         try:
#             c_batch, sample_lengths = pad(
#                 [s[data_type] for s in batch if s[data_type] is not None], pad_val
#             )
#             batch_out[data_type + "s"] = c_batch
#             batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
#         except Exception as e:
#             print(f"Error in collate_pad for {data_type}: {e}")
#             for i, s in enumerate(batch):
#                 print(f"Sample {i} - {data_type}: {s[data_type].shape if s[data_type] is not None else 'None'}")
#             raise e
#     return batch_out

def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        pad_val = -1 if data_type == "target" else 0.0
        try:
            c_batch, sample_lengths = pad(
                [s[data_type] for s in batch if s[data_type] is not None], pad_val
            )
            batch_out[data_type + "s"] = c_batch
            batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
        except Exception as e:
            print(f"Error in collate_pad for {data_type}: {e}")
            print(f"Batch keys: {list(batch[0].keys())}")
            for i, s in enumerate(batch):
                print(f"Sample {i} - {data_type}: {s[data_type].shape if s[data_type] is not None else 'None'}")
            raise e
    return batch_out

class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class DataModule(LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.cfg.gpus = torch.cuda.device_count()
        self.total_gpus = torch.cuda.device_count()
        
        # self.train_filenames = glob(f"/home/work/data/LRS2_YOLO/pretrain/*/*.pkl")
        # self.val_filenames = glob(f"/home/work/data/LRS2_YOLO/val_sample/*/*.pkl")
        # self.test_filenames = glob(f"/home/work/data/LRS2_YOLO/val_sample/*/*.pkl")
        self.train_filenames = glob(f"/home/work/data/aihub_final/train/*/*.pkl")
        self.val_filenames = glob(f"/home/work/data/aihub_final/val/*/*.pkl")
        self.test_filenames = glob(f"/home/work/data/aihub_final/main/*/*.pkl")


    def _dataloader(self, ds, collate_fn):
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        # 디버깅: 데이터 로드 테스트
        for idx, batch in enumerate(dataloader):
            print(f"Batch {idx}:")
            for key, value in batch.items():
                print(f"  {key}: {value.shape if isinstance(value, torch.Tensor) else 'N/A'}")
            if idx == 5:  # 처음 몇 개만 확인
                break
        return dataloader

    def train_dataloader(self):
        # 필터링된 파일 리스트로 데이터셋을 만듭니다.
        valid_train_filenames = [filename for filename in self.train_filenames if self._is_valid_file(filename)]
        train_ds = AVDataset(
            valid_train_filenames,
            modality=self.cfg.data.modality,
            audio_transform=None,
            video_transform=VideoTransform("train"),
            language=self.cfg.data.language
        )
        return self._dataloader(train_ds, collate_pad)

    def val_dataloader(self):
        # 필터링된 파일 리스트로 데이터셋을 만듭니다.
        valid_val_filenames = [filename for filename in self.val_filenames if self._is_valid_file(filename)]
        # val_ds = AVDataset(
        #     valid_val_filenames,
        #     modality=self.cfg.data.modality,
        #     audio_transform=None,
        #     video_transform=None,
        #     language=self.cfg.data.language
        # )
        val_ds = AVDataset(
            valid_val_filenames,
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("val"),
            video_transform=None, #VideoTransform("val"),
            language=self.cfg.data.language
        )
        return self._dataloader(val_ds, collate_pad)

    def test_dataloader(self):
        # 필터링된 파일 리스트로 데이터셋을 만듭니다.
        valid_test_filenames = [filename for filename in self.test_filenames if self._is_valid_file(filename)]
        dataset = AVDataset(
            valid_test_filenames,
            modality=self.cfg.data.modality,
            audio_transform=None,
            video_transform=None,
            language=self.cfg.data.language
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader

    def _is_valid_file(self, filename):
        """유효한 .pkl 파일인지 확인하는 함수"""
        try:
            data = torch.load(filename)
            # 파일 로딩 성공, True 반환
            return True
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return False
