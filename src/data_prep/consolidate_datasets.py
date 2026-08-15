"""Consolidates the 3 source datasets into data/unified/<class>/, deduping by file hash.

Usage: python3 src/data_prep/consolidate_datasets.py
"""
import hashlib
import imghdr
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from class_mapping import SOURCES, UNIFIED_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VALID_KINDS = {"jpeg", "png", "gif", "bmp"}


def file_hash(path, chunk_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path):
    """Rejects corrupt files and files whose real format (e.g. webp) doesn't match
    what tf.keras.utils.image_dataset_from_directory's decoder supports."""
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception:
        return False
    return imghdr.what(path) in VALID_KINDS


def main():
    if UNIFIED_DIR.exists():
        shutil.rmtree(UNIFIED_DIR)
    for cls in UNIFIED_CLASSES:
        (UNIFIED_DIR / cls).mkdir(parents=True, exist_ok=True)

    seen_hashes = defaultdict(set)  # unified_class -> set of md5 hashes already copied
    counts = defaultdict(lambda: defaultdict(int))  # source -> class -> count
    dupes = defaultdict(int)
    dropped = defaultdict(int)
    invalid = defaultdict(int)

    for source in SOURCES:
        root = PROJECT_ROOT / source["root"]
        if not root.exists():
            print(f"WARNING: source root not found, skipping: {root}")
            continue
        for src_class_dir in sorted(root.iterdir()):
            if not src_class_dir.is_dir():
                continue
            src_class_name = src_class_dir.name
            unified_class = source["class_map"].get(src_class_name, "__unmapped__")
            if unified_class is None:
                dropped[src_class_name] += sum(
                    1 for f in src_class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
                )
                continue
            if unified_class == "__unmapped__":
                print(f"WARNING: unmapped class '{src_class_name}' in {source['name']}, skipping")
                continue

            for img_path in sorted(src_class_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                if not is_valid_image(img_path):
                    invalid[unified_class] += 1
                    continue
                h = file_hash(img_path)
                if h in seen_hashes[unified_class]:
                    dupes[unified_class] += 1
                    continue
                seen_hashes[unified_class].add(h)

                dest_name = f"{source['name']}_{img_path.stem}{img_path.suffix.lower()}"
                dest_path = UNIFIED_DIR / unified_class / dest_name
                shutil.copy2(img_path, dest_path)
                counts[source["name"]][unified_class] += 1

    print("\n=== Copied per source per class ===")
    for source_name, cls_counts in counts.items():
        for cls, n in sorted(cls_counts.items()):
            print(f"  {source_name:12s} {cls:12s} {n}")

    print("\n=== Dropped classes (not in unified taxonomy) ===")
    for cls, n in sorted(dropped.items()):
        print(f"  {cls:12s} {n} images dropped")

    print("\n=== Duplicate images skipped (exact md5 match within class) ===")
    for cls, n in sorted(dupes.items()):
        print(f"  {cls:12s} {n} duplicates skipped")

    print("\n=== Invalid/corrupt images skipped ===")
    for cls, n in sorted(invalid.items()):
        print(f"  {cls:12s} {n} invalid images skipped")

    print("\n=== Final unified class counts ===")
    total = 0
    for cls in UNIFIED_CLASSES:
        n = len(list((UNIFIED_DIR / cls).iterdir()))
        total += n
        print(f"  {cls:12s} {n}")
    print(f"  {'TOTAL':12s} {total}")


if __name__ == "__main__":
    main()
