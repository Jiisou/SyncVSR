# encoding: utf-8
import os
from glob import glob
from tqdm import tqdm
import random
from multiprocessing import Pool, set_start_method
import cv2

import torch
from pydub import AudioSegment
from ultralytics import YOLO
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE

from utils import retrieve_txt

# constants
MODEL_PATH = "/home/work/SyncVSR/LRS/video/preprocess/yolov8n-face.pt" # https://github.com/akanametov/yolo-face?tab=readme-ov-file#trained-models
jpeg = TurboJPEG()

def load_model(model_path):
    """load model from path"""
    model = YOLO(model_path)
    return model

def find_bbox(image, model=YOLO(MODEL_PATH), device="cuda"):
    """find bounding box"""
    res = model.predict(
      image,
      save=False, 
      save_txt=False, 
      verbose=False, 
      imgsz=640, 
      device=device,
      max_det=1,
    )
    boxes = res[0].boxes  # first image
    boxes = boxes.cpu()
    return boxes.xywh.numpy()

def extract_yolov8(mp4_path):
    # open video
    cap = cv2.VideoCapture(mp4_path)
    video = []
    frame_idx = 0
    past_bbox = None
    while cap.isOpened():
        ret, frame = cap.read()  # BGR
        if ret:
            list_bbox = find_bbox(frame)
            if len(list_bbox) == 0 and frame_idx == 0:
                center_x, center_y, width, height = frame.shape[1] // 2, frame.shape[0] // 2, 128, 128
                past_bbox = [center_x, center_y, width, height]
            elif len(list_bbox) == 0 and frame_idx > 0:
                center_x, center_y, width, height = past_bbox
            else:
                first_bbox = list_bbox[0]
                center_x, center_y, width, height = first_bbox
                past_bbox = first_bbox


            # WARNING: THERE CAN BE MULTIPLE SIZES AT ONE VIDEO!
            crop_size = 128            
            border = crop_size // 2

            # push center_y down a bit
            center_y += 0.2 * height

            # take care of overflow
            if center_x < border:
                center_x = border
            if center_y < border:
                center_y = border
            if center_x > frame.shape[1] - border:
                center_x = frame.shape[1] - border
            if center_y > frame.shape[0] - border:
                center_y = frame.shape[0] - border
            # crop
            cropped_frame = frame[
                int(center_y - border) : int(center_y + border),
                int(center_x - border) : int(center_x + border),
            ]
            cropped_frame = jpeg.encode(cropped_frame)
            video.append(cropped_frame)
            frame_idx += 1
        else:
            break
    cap.release()
    return video
import subprocess

def has_audio(file_path):
    """Check if the file has an audio stream using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-i", file_path, "-show_streams", "-select_streams", "a", "-loglevel", "error"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return bool(result.stdout.strip())  # If ffprobe output is not empty, audio exists
    except Exception as e:
        print(f"FFPROBE ERR: {e}")
        return False

def preprocess(file_name):
    result = {}
    try:
        result["video"] = extract_yolov8(file_name)
    except Exception as e:
        print(f"YOLO ERR: {file_name} - {e}")
        return
    
    if has_audio(file_name):
        try:
            # audio_file_name = file_name.replace("_final", "_audio")
            result["audio"] = AudioSegment.from_file(file_name, format="mp4")
            # result["audio"] = AudioSegment.from_file(file_name, formal="wav")
        except Exception as e:
            print(f"AUDIO ERR: {file_name} - {e}")
            result["audio"] = None  # Handle files with unreadable audio
    else:
        print(f"No audio track found in {file_name}")
        result["audio"] = None

    try:
        result["text"] = retrieve_txt(file_name)
    except Exception as e:
        print(f"TEXT ERR: {file_name} - {e}")
        return

    # Save logic remains unchanged
    data_dir = "/home/work/data/aihub_preprocessed"
    target_dir = "/home/work/data/aihub_pkl"
    savename = file_name.replace(data_dir, target_dir).replace(".mp4", ".pkl")
    try:
        if not os.path.exists(os.path.dirname(savename)):
            os.makedirs(os.path.dirname(savename))  # Create target folder if it does not exist
    except Exception as e:
        print(f"DIR ERR: {file_name} - {e}")
        return
    
    try:
        assert len(result["video"]) > 0, f"YOLO ERR: {file_name} failed to process"
        if result["audio"] is not None:
            assert len(result["audio"]) > 0, f"AUDIO ERR: {file_name} failed to process"
    except Exception as e:
        print(e)
        return
    
    torch.save(result, savename)
    del result
    torch.cuda.empty_cache()
    return

# def preprocess(file_name):
#     result = {}
#     result["video"] = extract_yolov8(file_name)
#     result["audio"] = AudioSegment.from_file(file_name, format="mp4")
#     result["text"] = retrieve_txt(file_name)

#     # save
#     data_dir = "/home/work/data/aihub_preprocessed"
#     target_dir = "/home/work/data/aihub_pkl"
#     savename = file_name.replace(data_dir, target_dir).replace(".mp4", ".pkl")
#     try:
#         if not os.path.exists(os.path.dirname(savename)):
#             os.makedirs(os.path.dirname(savename)) # if the folder does not exist, create it
#     except:
#         print(f"DIR ERR: {file_name}")
#         return
#     try:
#         assert len(result["video"]) > 0, f"YOLO ERR: {file_name} failed to processed"
#         assert len(result["audio"]) > 0, f"AUDIO ERR: {file_name} failed to processed"
#     except:
#         return
    
#     torch.save(result, savename)
#     del result
#     torch.cuda.empty_cache()
#     return
    

if __name__ == "__main__":
    set_start_method('spawn')

    num_cpus = os.cpu_count()
    print("Number of cpus: {}".format(num_cpus))
    batch_size = num_cpus - 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        num_workers = num_cpus - 1
    elif device == "cpu":
        num_workers = 0

    data_dir = "/home/work/data/aihub_preprocessed"
    target_dir = "/home/work/data/aihub_pkl"

    # Fetch mp4s
    # train_mp4_files = glob(os.path.join(data_dir, "train", "*.mp4"))
    # main_mp4_files = glob(os.path.join(data_dir, "main", "*.mp4"))
    train_mp4_files = glob(os.path.join(data_dir, "train", "**", "*.mp4"))
    main_mp4_files = glob(os.path.join(data_dir, "main", "**", "*.mp4"))

    train_mp4_basename = [item.replace(data_dir, "") for item in train_mp4_files]
    train_mp4_basename = [item.replace(".mp4", "") for item in train_mp4_basename]
    main_mp4_basename = [item.replace(data_dir, "") for item in main_mp4_files]
    main_mp4_basename = [item.replace(".mp4", "") for item in main_mp4_basename]

    # Fetch pkls
    # train_pkl_files = glob(os.path.join(target_dir, "train",  "*.pkl"))
    # main_pkl_files = glob(os.path.join(target_dir, "main", "*.pkl"))
    train_pkl_files = glob(os.path.join(target_dir, "train", "**", "*.pkl"))
    main_pkl_files = glob(os.path.join(target_dir, "main", "**", "*.pkl"))

    train_pkl_basename = [item.replace(target_dir, "") for item in train_pkl_files]
    train_pkl_basename = [item.replace(".pkl", "") for item in train_pkl_basename]
    main_pkl_basename = [item.replace(target_dir, "") for item in main_pkl_files]
    main_pkl_basename = [item.replace(".pkl", "") for item in main_pkl_basename]
    
    # Include items only when it is not included in list_target_basename
    unique_train_items = list(set(train_mp4_basename) - set(train_pkl_basename))
    unique_main_items = list(set(main_mp4_basename) - set(main_pkl_basename))
    
    unique_items = unique_train_items + unique_main_items
    todo_process_files = [f"{data_dir}{item}.mp4" for item in unique_items]
    random.shuffle(todo_process_files)
    print(f"original mp4: {len(train_mp4_files+main_mp4_files)} -> todo process: {len(todo_process_files)}")

    with Pool(batch_size) as p:
        print("mapping ...")
        results = tqdm(p.imap(preprocess, todo_process_files), total=len(todo_process_files))
        print("running ...")
        list(results)  # fetch the lazy results
        print("done")
