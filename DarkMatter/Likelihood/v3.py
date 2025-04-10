#!/usr/bin/env python3
import argparse
import logging
import re
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def calculate_hash(filepath: Path) -> str:
    """Calculate the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with filepath.open('rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_hash(folder: Path) -> bool:
    """Verify file hashes against allfiles.hash in parallel."""
    hash_file = folder / "allfiles.hash"
    if not hash_file.is_file():
        logger.error(f"Hash file 'allfiles.hash' not found in {folder}")
        return False

    with hash_file.open('r') as f:
        hash_entries = [line.strip().split(maxsplit=1) for line in f if line.strip()]

    with ThreadPoolExecutor() as executor:
        futures = []
        for expected_hash, filename in hash_entries:
            if filename == "allfiles.hash":
                continue
            filepath = folder / filename
            if not filepath.is_file():
                logger.error(f"Missing file {filename} in {folder}")
                return False
            futures.append(executor.submit(
                lambda f=filepath, e=expected_hash: (f, e, calculate_hash(f))
            ))

        for future in futures:
            filepath, expected_hash, actual_hash = future.result()
            if expected_hash != actual_hash:
                logger.error(f"Hash mismatch for {filepath.name}: Expected {expected_hash}, Found {actual_hash}")
                return False

    logger.info(f"Hash verification passed for {folder}")
    return True

def find_matching_folders(base_dir: Path, patterns: list[str]) -> list[Path]:
    """Find directories matching patterns and verify their hashes."""
    valid_folders = []
    folders_found = False

    for pattern in patterns:
        logger.info(f"Searching for folders matching pattern: {pattern}")
        matching_dirs = [p for p in base_dir.glob(f"**/{pattern}") if p.is_dir()]
        
        if not matching_dirs:
            logger.info(f"No folders found for pattern: {pattern}")
            continue
        folders_found = True

        for folder in matching_dirs:
            logger.info(f"Processing {folder}")
            if (folder / "tmp").is_dir():
                logger.info(f"Skipping hash verification for {folder} due to 'tmp' folder")
                valid_folders.append(folder)
            elif verify_hash(folder):
                valid_folders.append(folder)
            else:
                logger.error(f"Hash verification failed for {folder}")

    if not folders_found:
        logger.error("No folders matched any provided patterns")
        raise ValueError("No matching folders found")
    return valid_folders

def move_folders(valid_folders: list[Path]) -> None:
    """Move verified folders to target directories."""
    for folder in valid_folders:
        match = re.search(r'7DT[0-9]{2}', str(folder))
        if not match:
            logger.error(f"Unable to find key '7DT??' in {folder}. Skipping.")
            continue
        key = match.group(0)
        target_dir = Path(f"/data/data1/obsdata/{key}")
        
        if not target_dir.is_dir():
            logger.error(f"Target directory {target_dir} does not exist. Skipping {folder}")
            continue

        try:
            shutil.move(str(folder), str(target_dir))
            logger.info(f"Moved {folder} to {target_dir}")
        except Exception as e:
            logger.error(f"Failed to move {folder}: {e}")

def main() -> None:
    """Main function to parse arguments and execute the workflow."""
    parser = argparse.ArgumentParser(description="Verify and move directories based on hash files.")
    parser.add_argument("patterns", nargs="+", help="Folder name patterns to search for (e.g., '*test*')")
    args = parser.parse_args()

    base_dir = Path("/data/data1/obsdata/obsdata_from_mcs/")
    try:
        valid_folders = find_matching_folders(base_dir, args.patterns)
        #move_folders(valid_folders)
        logger.info("All matching folders processed successfully")
    except ValueError as e:
        logger.error(str(e))
        exit(1)

if __name__ == "__main__":
    main()