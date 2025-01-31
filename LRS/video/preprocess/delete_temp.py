import os
from pathlib import Path

def remove_files_with_suffix(directory, suffixes):
    """
    Remove files in a directory (and its subdirectories) that have specific suffixes.

    :param directory: Path to the directory to clean.
    :param suffixes: List of suffixes to match files for removal.
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return

    for file in directory.rglob("*"):  # Recursively find all files
        if file.is_file() and any(file.name.endswith(suffix) for suffix in suffixes):
            try:
                os.remove(file)
                print(f"Deleted file: {file}")
            except Exception as e:
                print(f"Failed to delete file {file}: {e}")

if __name__ == "__main__":
    # Define the directory to search and suffixes to remove
    # target_directory = "/home/work/data/ko/02_vid_after_preprocess"
    target_directory = "/home/work/data/ko/0_after_preprocessed"
    suffixes_to_remove = ["_audio.wav", "_video.mp4"]

    # Call the function to remove files
    remove_files_with_suffix(target_directory, suffixes_to_remove)
