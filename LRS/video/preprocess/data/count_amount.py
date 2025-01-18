import os

def count_pkl_files_in_directory(directory_path):
    """
    Count the number of .pkl files in a given directory and its subdirectories.

    Parameters:
        directory_path (str): The root directory to start the search.

    Returns:
        int: The total number of .pkl files.
    """
    pkl_count = 0

    # Walk through the directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_count += 1

    return pkl_count

# Example usage
# directory_path = input("Enter the directory path: ")
dir_path = "./preprocess/data/mvlrs_v1_YOLO/main"
if os.path.exists(dir_path):
    total_pkl_files = count_pkl_files_in_directory(dir_path)
    print(f"Total .pkl files: {total_pkl_files}")
else:
    print("The provided directory path does not exist.")
