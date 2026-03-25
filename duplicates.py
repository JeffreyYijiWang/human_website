import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

# Set your source directory
BASE_DIR = Path("images/female")
# Set where the duplicates should go
DUP_DIR = Path("female_duplicates")

def get_file_hash(file_path):
    """Generates an MD5 hash for a file using chunks to save RAM."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Could not hash {file_path}: {e}")
        return None

def move_duplicates(root_path, target_path):
    # Ensure the duplicates folder exists
    target_path.mkdir(parents=True, exist_ok=True)
    
    hashes = defaultdict(list)
    files = [f for f in root_path.rglob("*") if f.is_file()]
    
    print(f"Scanning {len(files)} files for duplicates...")

    for path in files:
        # Skip files already in the duplicates folder if it's inside BASE_DIR
        if target_path in path.parents:
            continue
            
        file_hash = get_file_hash(path)
        if file_hash:
            hashes[file_hash].append(path)

    moved_count = 0
    for file_hash, paths in hashes.items():
        if len(paths) > 1:
            # We keep the FIRST one found (paths[0])
            # And move all others (paths[1:])
            original = paths[0]
            for duplicate_path in paths[1:]:
                # Construct new path: duplicates/original_filename_hash.ext
                # We add the hash to the name to prevent overwriting if 
                # different duplicates have the same filename
                new_name = f"{duplicate_path.stem}_{file_hash[:8]}{duplicate_path.suffix}"
                destination = target_path / new_name
                
                print(f"Moving: {duplicate_path.name} -> {destination}")
                shutil.move(str(duplicate_path), str(destination))
                moved_count += 1

    print(f"\n✅ Done! Moved {moved_count} duplicate files to '{target_path}'.")

if __name__ == "__main__":
    if BASE_DIR.exists():
        move_duplicates(BASE_DIR, DUP_DIR)
    else:
        print(f"Error: The directory '{BASE_DIR}' was not found.")