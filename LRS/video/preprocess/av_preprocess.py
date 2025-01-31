import os
from pathlib import Path
import ffmpeg
from moviepy.editor import VideoFileClip, AudioFileClip

def downsample_audio(input_mp4, output_audio, target_sample_rate=16000):
    if not os.path.exists(input_mp4):
        raise FileNotFoundError(f"Input file does not exist: {input_mp4}")
    ffmpeg.input(input_mp4).output(
        output_audio, ar=target_sample_rate, ac=1, format="wav"
    ).overwrite_output().run()
    print(f"Audio downsampled and saved to {output_audio}")

def process_all_videos(input_directory, output_directory):
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for mp4_file in input_directory.rglob("*.mp4"):
        print(f"Processing: {mp4_file}")

        try:
            # Create output path preserving the directory structure
            relative_path = mp4_file.relative_to(input_directory)
            temp_path = output_directory / relative_path.parent
            final_output = output_directory / relative_path

            # Ensure the temp directory exists
            os.makedirs(temp_path, exist_ok=True)

            output_audio = temp_path / (relative_path.stem + "_audio.wav")
            output_video = temp_path / (relative_path.stem + "_video.mp4")

            # 오디오 다운샘플링
            downsample_audio(str(mp4_file), str(output_audio))

            # 비디오 처리
            process_video_with_center_crop(str(mp4_file), str(output_video))

            # 동기화 및 최종 MP4 생성
            combine_audio_video(str(output_video), str(output_audio), str(final_output))

            # 샘플링 비율 고정
            fixed_output = final_output.with_name(final_output.stem + "_fixed.mp4")
            fix_audio_sample_rate(final_output, fixed_output)

            # 최종 결과를 덮어쓰기
            os.replace(fixed_output, final_output)

        except Exception as e:
            print(f"Error processing {mp4_file}: {e}")


def process_video_with_center_crop(input_mp4, output_video, target_fps=25, target_size=160):
    ffmpeg.input(input_mp4).filter("fps", fps=target_fps).filter(
        "crop",
        f"min(iw,ih)",
        f"min(iw,ih)",
        f"(iw-min(iw,ih))/2",
        f"(ih-min(iw,ih))/2"
    ).filter(
        "scale", width=target_size, height=target_size
    ).output(output_video, vcodec="libx264").overwrite_output().run()
    print(f"Video processed with center crop and saved to {output_video}")

def combine_audio_video(video_path, audio_path, output_mp4, audio_sample_rate=16000):
    temp_audio_path = str(audio_path).replace(".wav", "_resampled.wav")

    ffmpeg.input(audio_path).output(
        temp_audio_path, ar=audio_sample_rate, ac=1
    ).overwrite_output().run()

    video_clip = VideoFileClip(str(video_path))
    resampled_audio_clip = AudioFileClip(temp_audio_path)
    final_clip = video_clip.set_audio(resampled_audio_clip)
    final_clip.write_videofile(str(output_mp4), codec="libx264", audio_codec="aac")

    os.remove(temp_audio_path)
    print(f"Final MP4 created at {output_mp4} with audio sample rate {audio_sample_rate}")

def fix_audio_sample_rate(input_mp4, output_mp4, target_sample_rate=16000):
    ffmpeg.input(input_mp4).output(
        output_mp4,
        ar=target_sample_rate,
        ac=1,
        vcodec="copy",
        codec="aac"
    ).overwrite_output().run()
    print(f"Fixed audio sample rate to {target_sample_rate}Hz and saved to {output_mp4}")

if __name__ == "__main__":
    input_directory = "/home/work/data/ko/0_clip_before_preprocess"
    output_directory = "/home/work/data/ko/0_after_preprocessed"
    # input_directory = "/home/work/data/ko/TS104/소음환경3/C(일반인)/M(남성)/M(남성)_4"
    # output_directory = "/home/work/data/ko/02_vid_after_preprocess"

    process_all_videos(input_directory, output_directory)
