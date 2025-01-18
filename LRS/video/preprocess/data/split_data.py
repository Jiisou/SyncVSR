import os
from tqdm import tqdm

def split_pkl_files_by_folders(base_path, test_path, val_path):
    """
    Split .pkl files into test and val sets based on folder count and create corresponding directories.

    Parameters:
        base_path (str): The base directory containing .pkl files in subfolders.
        test_path (str): The directory to store test set .pkl files.
        val_path (str): The directory to store val set .pkl files.

    Returns:
        dict: A dictionary with 'test' and 'val' keys containing respective .pkl file paths.
    """
    # Collect all folders containing .pkl files
    folder_paths = []
    for root, _, files in os.walk(base_path):
        if any(file.endswith('.pkl') for file in files):
            folder_paths.append(root)

    # Sort the folders to ensure consistent order
    folder_paths.sort()

    # Determine split point
    total_folders = len(folder_paths)
    split_point = total_folders // 2

    # Split folders into test and val sets
    test_folders = folder_paths[:split_point]
    val_folders = folder_paths[split_point:]

    # Create directories for test and val sets
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # Collect .pkl files for each set with tqdm progress bar
    split_data = {'test': [], 'val': []}

    print("Processing test folders:")
    for folder in tqdm(test_folders, desc="Test folders"):
        for file in os.listdir(folder):
            if file.endswith('.pkl'):
                src_path = os.path.join(folder, file)
                dest_path = os.path.join(test_path, file)
                os.rename(src_path, dest_path)
                split_data['test'].append(dest_path)

    print("Processing val folders:")
    for folder in tqdm(val_folders, desc="Val folders"):
        for file in os.listdir(folder):
            if file.endswith('.pkl'):
                src_path = os.path.join(folder, file)
                dest_path = os.path.join(val_path, file)
                os.rename(src_path, dest_path)
                split_data['val'].append(dest_path)

    return split_data

# Example usage
base_path = "./preprocess/data/mvlrs_v1_YOLO/main"
test_path = "./preprocess/data/test"
val_path = "./preprocess/data/val"
split_result = split_pkl_files_by_folders(base_path, test_path, val_path)
# print("Test files:", split_result['test'])
# print("Val files:", split_result['val'])

