import numpy as np
from typing import Tuple
import pydub

def retrieve_txt(path):
    # replace .mp4 with .txt
    path = path.replace(".mp4", ".txt")
    # read text
    with open(path, "r") as f:
        text = f.read()
    return text

# def pydub_to_np(audio: pydub.AudioSegment) -> Tuple[np.ndarray, int]:
#     """
#     https://stackoverflow.com/a/66922265/8380469
#     Converts pydub audio segment into np.float32 of shape [channels ,duration_in_seconds*sample_rate],

#     where each value is in range [-1.0, 1.0]. 

#     Returns tuple (audio_np_array, sample_rate).
#     """
#     # return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((audio.channels, -1)) / (
#     #         1 << (8 * audio.sample_width - 1)), audio.frame_rate
#     print(f"Before Converting AudioSegment: channels={audio.channels}, sampling_rate={audio.frame_rate}, "
#         f"sample_width={audio.sample_width}, duration={len(audio)} ms")
#     samples = audio.get_array_of_samples()
#     print(f"Number of samples: {len(samples)}")

#     audio_np = np.array(samples, dtype=np.float32).reshape((audio.channels, -1)) / (
#         1 << (8 * audio.sample_width - 1)), audio.frame_rate
#     print(audio_np)
#     return audio_np
#     # return np.array(samples, dtype=np.float32).reshape((audio.channels, -1)) / (
#     #     1 << (8 * audio.sample_width - 1)), audio.frame_rate


# def pydub_to_np(audio: AudioSegment) -> Tuple[np.ndarray, int]:
#     """
#     Converts pydub audio segment into np.float32 of shape [channels, duration_in_seconds*sample_rate],
#     where each value is in range [-1.0, 1.0].
    
#     Returns tuple (audio_np_array, sample_rate).
#     """
#     # 모노(1채널)로 변환
#     if audio.channels > 1:
#         print(f"Input audio has {audio.channels} channels. Converting to mono...")
#         audio = audio.set_channels(1)

#     print(f"Before Converting AudioSegment: channels={audio.channels}, sampling_rate={audio.frame_rate}, "
#           f"sample_width={audio.sample_width}, duration={len(audio)} ms")
#     samples = audio.get_array_of_samples()
#     print(f"Number of samples: {len(samples)}")

#     audio_np = np.array(samples, dtype=np.float32) / (1 << (8 * audio.sample_width - 1))
#     print(f"Converted to numpy array of shape: {audio_np.shape}, dtype: {audio_np.dtype}")
#     return audio_np, audio.frame_rate

from pydub import AudioSegment
import numpy as np
from typing import Tuple

def pydub_to_np(audio: AudioSegment, target_sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Converts pydub audio segment into np.float32 of shape [channels, duration_in_seconds*sample_rate],
    where each value is in range [-1.0, 1.0]. Also resamples the audio to target_sample_rate.

    Args:
        audio (AudioSegment): Input audio segment.
        target_sample_rate (int, optional): The target sampling rate. Defaults to 16000.

    Returns:
        Tuple[np.ndarray, int]: Converted numpy array and the new sample rate.
    """
    # 1. 모노(1채널)로 변환
    if audio.channels > 1:
        print(f"Input audio has {audio.channels} channels. Converting to mono...")
        audio = audio.set_channels(1)

    # 2. 샘플링 레이트 변경 (44,100Hz -> 16,000Hz)
    if audio.frame_rate != target_sample_rate:
        print(f"Resampling audio from {audio.frame_rate} Hz to {target_sample_rate} Hz...")
        audio = audio.set_frame_rate(target_sample_rate)

    print(f"After Processing: channels={audio.channels}, sampling_rate={audio.frame_rate}, "
          f"sample_width={audio.sample_width}, duration={len(audio)} ms")

    # 3. 오디오 샘플을 numpy 배열로 변환
    samples = audio.get_array_of_samples()
    print(f"Number of samples: {len(samples)}")

    audio_np = np.array(samples, dtype=np.float32) / (1 << (8 * audio.sample_width - 1))
    print(f"Converted to numpy array of shape: {audio_np.shape}, dtype: {audio_np.dtype}")

    return audio_np, audio.frame_rate
