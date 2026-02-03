import kagglehub
import os
import shutil

# Target directory
DATA_DIR = os.path.join(os.getcwd(), "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def download_and_move(dataset_handle, target_files_map):
    """
    Downloads a dataset and moves specific files to data/ diretory.
    target_files_map: dict where key is current filename (substring match), value is new filename
    """
    print(f"Downloading {dataset_handle}...")
    try:
        # Download latest version
        path = kagglehub.dataset_download(dataset_handle)
        print(f"Downloaded to cache: {path}")

        files = os.listdir(path)
        print(f"Files found in dataset: {files}")

        found_any = False
        for search_str, new_name in target_files_map.items():
            # Find file that contains the search string
            match = next((f for f in files if search_str in f), None)
            
            if match:
                source_path = os.path.join(path, match)
                dest_path = os.path.join(DATA_DIR, new_name)
                print(f"Moving {match} -> {dest_path}")
                shutil.copy(source_path, dest_path)
                found_any = True
            else:
                print(f"WARNING: Could not find file matching '{search_str}' in {files}")

        if not found_any:
            print("Fallback: Copying all CSVs...")
            for f in files:
                if f.endswith(".csv"):
                    shutil.copy(os.path.join(path, f), DATA_DIR)

    except Exception as e:
        print(f"Error downloading {dataset_handle}: {e}")

if __name__ == "__main__":
    print("Starting automated download...")
    
    # 1. Primary Dataset (File usually named something like PS_2017...csv)
    # We rename it to onlinefraud.csv
    download_and_move("rupakroy/online-payments-fraud-detection-dataset", {
        ".csv": "onlinefraud.csv" 
    })
    
    # 2. Secondary Dataset (Sparkov)
    # We want both Test and Train files
    download_and_move("kartik2112/fraud-detection", {
        "fraudTest.csv": "fraudTest.csv",
        "fraudTrain.csv": "fraudTrain.csv"
    })
    
    print("\nDownload complete. Please check the 'data/' folder.")
