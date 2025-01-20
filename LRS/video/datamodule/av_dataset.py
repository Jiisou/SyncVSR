import random
import numpy as np

import torch
import torchvision
from turbojpeg import TJPF_GRAY, TurboJPEG

from preprocess.utils import pydub_to_np
from .transforms import TextTransform, FunctionalModule


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filenames,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
        language=None,
    ):
        self.filenames = filenames
        self.jpeg = TurboJPEG()

        self.modality = modality
        self.rate_ratio = rate_ratio

        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.lrs3_video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.Resize(96),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize(0.421, 0.165),
        )
        self.lrs2_video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.CenterCrop(96),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

        self.length = np.load("./datamodule/video_length.npy")
        self.length_distribution = np.bincount(self.length)
        self.cut = self.length.max()
        self.tokenizer = TextTransform(
            sp_model_path="./spm/unigram/unigram5000.model",
            dict_path="./spm/unigram/unigram5000_units.txt",
        )
        self.video_max_length = self.length.max()
        self.language = language
        self.probability_distribution = self.length_distribution/self.length_distribution.sum()
        self.neural_audio_codec_sample_rate = 16000
        self.audio_padding_max = int((self.video_max_length / 25) * self.neural_audio_codec_sample_rate)
        
        # multiple for video-audio alignment
        self.audio_multiple = 40
        print(f"using {self.language} with {self.cut} cut")

    # def __getitem__(self, idx):
    #     filename = self.filenames[idx]
    #     try:
    #         data = torch.load(filename)
    #     except Exception as e:
    #         print(f"Error loading {filename}: {e}")
    #         return None

    #     video = data["video"]
    #     audio = data["audio"]
    #     text = data["text"]

    #     # Debugging: Original lengths
    #     print(f"Original Sample {idx}: video={len(video)}, audio={len(audio)}")

    #     # Convert audio to numpy array
    #     audio, sampling_rate = pydub_to_np(audio)
    #     print(f"pydub_to_np output: audio length = {len(audio)}, type = {type(audio)}")

    #     # Ensure audio is valid and not too short
    #     if not isinstance(audio, np.ndarray) or len(audio) < 10:
    #         print(f"Warning: Invalid or short audio for sample {idx}. Skipping.")
    #         return None

    #     # Adjust audio length
    #     target_audio_length = len(video) * self.audio_multiple
    #     audio = torch.as_tensor(audio, dtype=torch.float32).squeeze()

    #     if len(audio) < target_audio_length:
    #         padding_length = target_audio_length - len(audio)
    #         audio = torch.nn.functional.pad(audio, (0, padding_length))
    #     elif len(audio) > target_audio_length:
    #         audio = audio[:target_audio_length]

    #     # Debugging
    #     print(f"Adjusted Sample {idx}: video={len(video)}, audio={len(audio)}, ratio={len(audio) // len(video)}")

    #     # Process video
    #     video = np.stack([self.jpeg.decode(img, TJPF_GRAY) for img in video])
    #     video = torch.as_tensor(video, dtype=torch.float32).permute(0, 3, 1, 2)

    #     if self.video_transform:
    #         video = self.video_transform(video)

    #     # Tokenize text
    #     token_id = self.tokenizer.tokenize(text)

    #     return {"input": video, "audio": audio, "target": token_id}


    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # data = torch.load(filename)
        try:
            data = torch.load(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None

        # load from dictionary
        video = data["video"]
        video_size = len(video)
        audio = data["audio"]
        text = data["text"]
        text_data = text.split("\n")

        # 샘플 크기 확인
        print(f"Sample {idx}: video={len(video)}, audio={len(audio)}, text={text[:50]}")


        if video_size > self.cut: # pretrain or vox2
            # shard the text
            margin = 0.5 if "LRS3" in filename else 0.2

            # get random starting point and crop video and audio
            random_start = random.randint(0, video_size - self.cut)
            sample_length = np.random.choice(len(self.length_distribution), p=self.probability_distribution)
            video = video[random_start:random_start+sample_length]
            audio = audio[random_start*self.audio_multiple:(random_start+sample_length)*self.audio_multiple]

            # parse text for pretrain files with timestamps
            words = [text_data[i].split(" ") for i in range(4, len(text_data)-1)]
            
            # cut and align video, text and audio
            time_start = random_start / 25
            time_end = time_start + (sample_length / 25)
            
            list_word = []
            for word in words:
                if (time_start - margin) <= float(word[1]) and float(word[2]) < (time_end + margin):
                    list_word.append(word[0])
            
            text_string = " ".join(list_word)
    
        else: # main or short vox2
            text_string = text_data[0][5:].strip()
        
        token_id = self.tokenizer.tokenize(text_string)

        # post processing and transforming into tensor
        video = np.stack([self.jpeg.decode(img, TJPF_GRAY) for img in video])
        video = torch.as_tensor(video).permute(0, 3, 1, 2) # T x C x H x W
        video = video[:, :, :96, :96]  # (T, C, 96, 96)으로 강제 크기 고정

        if self.video_transform:
            video = self.video_transform(video)
        else:
            if "LRS3" in filename:
                video = self.lrs3_video_pipeline(video)
            elif "LRS2" in filename:
                video = self.lrs2_video_pipeline(video)

        # load audio
        audio, sampling_rate = pydub_to_np(audio)
        audio = torch.as_tensor(audio)
        audio = audio.squeeze()

        if len(audio) < len(video) * self.audio_multiple:
            print(filename, len(audio), len(video))
            padding_length = len(video) * self.audio_multiple - len(audio) #added
            audio = torch.nn.functional.pad(audio, (0, padding_length))    #added

        return {"input": video, "audio": audio, "target": token_id}

    def __len__(self):
        return len(self.filenames)
