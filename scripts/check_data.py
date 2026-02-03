import os

DATA_DIR = "../data"
REQUIRED_FILES = [
    "onlinefraud.csv",
    "fraudTest.csv", 
    "fraudTrain.csv"
]

def check_files():
    print("Checking for datasets...")
    missing = []
    for f in REQUIRED_FILES:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            print(f"[OK] Found {f}")
        else:
            print(f"[MISSING] {f}")
            missing.append(f)
    
    if missing:
        print("\nERROR: Missing datasets!")
        print("Please download the following files and place them in the 'data/' directory:")
        for m in missing:
            print(f" - {m}")
        print("\nRefer to data/README.md for download links.")
        exit(1)
    else:
        print("\nSUCCESS: All datasets found. You can proceed with analysis.")
        exit(0)

if __name__ == "__main__":
    check_files()
