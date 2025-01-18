import os
import whisper  # pip install openai-whisper
from pydub import AudioSegment

# 1. 오디오 추출 함수
def extract_audio_with_pydub(video_path, audio_output):
    """MP4에서 오디오 추출"""
    try:
        audio = AudioSegment.from_file(video_path, format="mp4")
        audio.export(audio_output, format="wav")
        print(f"Audio extracted to {audio_output}")
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False  # 실패 시 False 반환
    return True  # 성공 시 True 반환

# 2. Whisper로 음성 텍스트 변환 (GPU 활성화)
def transcribe_with_whisper(audio_path, model_name="large"):
    """Whisper로 음성을 텍스트로 변환"""
    try:
        print(f"Loading Whisper model: {model_name} (GPU enabled)")
        model = whisper.load_model(model_name, device="cuda")  # GPU를 사용하기 위해 `device="cuda"`
        result = model.transcribe(audio_path, word_timestamps=True)
    except Exception as e:
        print(f"Error transcribing audio {audio_path}: {e}")
        return None  # 실패 시 None 반환
    return result

# 3. 결과 포맷 변환 (절대적 타임스탬프 사용)
def format_transcription_to_custom_output(transcription_result):
    """Whisper 결과를 사용자 지정 형식으로 변환"""
    if transcription_result is None:
        return "Error during transcription.\n"
    
    output = []
    text = transcription_result.get("text", "").strip()
    segments = transcription_result.get("segments", [])

    # Text 추가
    output.append(f"Text:  {text}\n")

    # WORD START END 추가 (절대적 타임스탬프)
    output.append("WORD START END")
    for segment in segments:
        for word_info in segment["words"]:
            word = word_info["word"]
            start = round(word_info["start"], 2)
            end = round(word_info["end"], 2)
            output.append(f"{word} {start} {end}")
    return "\n".join(output)

# 4. 결과 저장
def save_to_text_file(data, output_path):
    """결과를 텍스트 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(data)
    print(f"Saved transcription to {output_path}")

# 5. 메인 처리 함수
def process_video(input_dir, model_name="large", resume_from=None):
    """MP4 파일 처리 및 결과 저장"""
    failed_files = []  # 에러가 발생한 파일 목록 저장
    process_next = False  # 다음 파일부터 시작할지 여부 결정

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                audio_path = os.path.join(root, f"{base_name}.wav")
                output_path = os.path.join(root, f"{base_name}.txt")

                # 중단된 위치 이후부터 실행
                if resume_from and video_path < resume_from:
                    continue

                print(f"Processing: {video_path}")

                # 오디오 추출
                if not extract_audio_with_pydub(video_path, audio_path):
                    failed_files.append(video_path)  # 오디오 추출 실패한 파일 기록
                    continue

                # Whisper로 변환
                transcription_result = transcribe_with_whisper(audio_path, model_name)
                if transcription_result is None:
                    failed_files.append(video_path)  # 변환 실패한 파일 기록
                    continue

                # 사용자 지정 형식으로 변환
                formatted_output = format_transcription_to_custom_output(transcription_result)

                # 결과 저장
                save_to_text_file(formatted_output, output_path)

                # 임시 오디오 파일 삭제
                if os.path.exists(audio_path):
                    os.remove(audio_path)

                # 중단된 파일을 기억
                resume_from = video_path  # 마지막으로 처리한 파일을 기억

    # 모든 처리가 끝난 후, 실패한 파일 목록 출력
    if failed_files:
        print("\nThe following files failed during processing:")
        for failed_file in failed_files:
            print(failed_file)
    else:
        print("\nAll files processed successfully.")

    # 중단된 파일 위치 반환
    return resume_from

# 실행
if __name__ == "__main__":
    input_directory = "/data/ko/clips10"  # MP4 파일이 있는 루트 디렉토리
    resume_file = '/data/ko/clips10/lip_J_1_F_03_C293_A_002/lip_J_1_F_03_C293_A_002_33.mp4'  # 중단된 파일 경로
    resume_from = resume_file  # 중단된 파일이 있으면 그 지점부터 시작하도록 설정 (clips10/Audio extracted to /data/ko/clips10/lip_J_1_F_03_C293_A_002/lip_J_1_F_03_C293_A_002_33.wav)
    resume_from = process_video(input_directory, model_name="large", resume_from=resume_from) # resume_from=None
