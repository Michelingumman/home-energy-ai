import os
import shutil
import gdown

# URL of your Google Drive folder containing the large files
drive_url = "https://drive.google.com/drive/folders/17CUy_SQne9ARlmMYxPeNBHUiWm434RLN?usp=sharing"
download_dir = "largeFiles"  # local folder where the drive folder will be downloaded

# Create the download directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

print("Downloading large files from Google Drive...")
# This downloads the entire folder (including subdirectories) into "largeFiles"
gdown.download_folder(drive_url, output=download_dir, quiet=False, use_cookies=False)

# Define mappings for where files should be placed
# These are based on the GitHub error messages showing both locations
mapping = {
    # Source paths in Google Drive -> Target paths in the project
    
    # X_test.npy files
    # We'll search for these files in both raw and processed folders from Google Drive
    os.path.join("raw", "X_test.npy"): [
        os.path.join("models", "test_data", "X_test.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "X_test.npy")
    ],
    os.path.join("processed", "X_test.npy"): [
        os.path.join("models", "test_data", "X_test.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "X_test.npy")
    ],
    
    # X_val.npy files
    os.path.join("raw", "X_val.npy"): [
        os.path.join("models", "test_data", "X_val.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "X_val.npy")
    ],
    os.path.join("processed", "X_val.npy"): [
        os.path.join("models", "test_data", "X_val.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "X_val.npy")
    ],
    
    # y_test.npy files
    os.path.join("raw", "y_test.npy"): [
        os.path.join("models", "test_data", "y_test.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "y_test.npy")
    ],
    os.path.join("processed", "y_test.npy"): [
        os.path.join("models", "test_data", "y_test.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "y_test.npy")
    ],
    
    # y_val.npy files
    os.path.join("raw", "y_val.npy"): [
        os.path.join("models", "test_data", "y_val.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "y_val.npy")
    ],
    os.path.join("processed", "y_val.npy"): [
        os.path.join("models", "test_data", "y_val.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "y_val.npy")
    ],
    
    # Timestamp files
    os.path.join("raw", "test_timestamps.npy"): [
        os.path.join("models", "test_data", "test_timestamps.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "test_timestamps.npy")
    ],
    os.path.join("processed", "test_timestamps.npy"): [
        os.path.join("models", "test_data", "test_timestamps.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "test_timestamps.npy")
    ],
    
    os.path.join("raw", "val_timestamps.npy"): [
        os.path.join("models", "test_data", "val_timestamps.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "val_timestamps.npy")
    ],
    os.path.join("processed", "val_timestamps.npy"): [
        os.path.join("models", "test_data", "val_timestamps.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "val_timestamps.npy")
    ],
}

# Also check if files are in the root of the downloaded folder
root_files = {
    "X_test.npy": [
        os.path.join("models", "test_data", "X_test.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "X_test.npy")
    ],
    "X_val.npy": [
        os.path.join("models", "test_data", "X_val.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "X_val.npy")
    ],
    "y_test.npy": [
        os.path.join("models", "test_data", "y_test.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "y_test.npy")
    ],
    "y_val.npy": [
        os.path.join("models", "test_data", "y_val.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "y_val.npy")
    ],
    "test_timestamps.npy": [
        os.path.join("models", "test_data", "test_timestamps.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "test_timestamps.npy")
    ],
    "val_timestamps.npy": [
        os.path.join("models", "test_data", "val_timestamps.npy"),
        os.path.join("src", "predictions", "prices", "models", "evaluation", "test_data", "val_timestamps.npy")
    ]
}

# First check for files in the specified subfolders
for src_relative, targets in mapping.items():
    src_path = os.path.join(download_dir, src_relative)
    if os.path.exists(src_path):
        print(f"Found file: {src_path}")
        for target in targets:
            # Ensure the target directory exists
            target_dir = os.path.dirname(target)
            os.makedirs(target_dir, exist_ok=True)
            # Copy the file (preserving metadata)
            print(f"Copying to {target}...")
            shutil.copy2(src_path, target)
            print(f"Successfully copied {src_path} to {target}")
    else:
        print(f"File {src_path} not found in the downloaded folder.")

# Then check for files directly in the root of the download directory
for root_file, targets in root_files.items():
    src_path = os.path.join(download_dir, root_file)
    if os.path.exists(src_path):
        print(f"Found file in root: {src_path}")
        for target in targets:
            # Ensure the target directory exists
            target_dir = os.path.dirname(target)
            os.makedirs(target_dir, exist_ok=True)
            # Copy the file (preserving metadata)
            print(f"Copying to {target}...")
            shutil.copy2(src_path, target)
            print(f"Successfully copied {src_path} to {target}")
    else:
        print(f"File {src_path} not found in the root of downloaded folder.")

print("Download and setup complete. Files have been placed in both locations mentioned in the GitHub errors.")
