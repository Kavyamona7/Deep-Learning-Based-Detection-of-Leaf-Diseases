from pathlib import Path
import hashlib
import random
import shutil

CLASSES = ("bacterial", "fungal")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def reset_split_structure(data_dir: Path) -> Path:
    split_dir = data_dir / "split"
    if split_dir.exists():
        shutil.rmtree(split_dir)

    for split_name in SPLITS:
        for class_name in CLASSES:
            (split_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

    return split_dir


def get_class_images(class_dir: Path):
    if not class_dir.is_dir():
        return []
    return sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def sha1_of_file(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.stem}__{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def build_targets_by_class(images_by_class):
    targets = {}
    for class_name in CLASSES:
        total = len(images_by_class[class_name])
        n_train = int(total * TRAIN_RATIO)
        n_val = int(total * VAL_RATIO)
        n_test = total - n_train - n_val
        targets[class_name] = {"train": n_train, "val": n_val, "test": n_test}
    return targets


def build_hash_groups(images_by_class):
    hash_groups = {}
    for class_name in CLASSES:
        for path in images_by_class[class_name]:
            digest = sha1_of_file(path)
            if digest not in hash_groups:
                hash_groups[digest] = []
            hash_groups[digest].append((class_name, path))
    return hash_groups


def assignment_cost(current_counts, targets, candidate_split, group_items):
    projected = {
        class_name: {split_name: current_counts[class_name][split_name] for split_name in SPLITS}
        for class_name in CLASSES
    }

    for class_name, _ in group_items:
        projected[class_name][candidate_split] += 1

    # Minimize total absolute deviation from 70/15/15 targets per class.
    cost = 0
    for class_name in CLASSES:
        for split_name in SPLITS:
            cost += abs(projected[class_name][split_name] - targets[class_name][split_name])
    return cost


def assign_groups_to_splits(hash_groups, targets):
    rng = random.Random(SEED)
    digests = list(hash_groups.keys())
    rng.shuffle(digests)

    current_counts = {
        class_name: {split_name: 0 for split_name in SPLITS} for class_name in CLASSES
    }
    digest_to_split = {}

    for digest in digests:
        group_items = hash_groups[digest]
        best_split = None
        best_cost = None

        for candidate_split in SPLITS:
            cost = assignment_cost(current_counts, targets, candidate_split, group_items)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_split = candidate_split

        digest_to_split[digest] = best_split
        for class_name, _ in group_items:
            current_counts[class_name][best_split] += 1

    return digest_to_split, current_counts


def copy_files_to_split(split_dir: Path, hash_groups, digest_to_split):
    for digest, items in hash_groups.items():
        split_name = digest_to_split[digest]
        for class_name, src in items:
            target = split_dir / split_name / class_name / src.name
            target = unique_target(target)
            shutil.copy2(src, target)


def summarize_split_counts(split_dir: Path):
    summary = {}
    for class_name in CLASSES:
        summary[class_name] = {}
        for split_name in SPLITS:
            p = split_dir / split_name / class_name
            summary[class_name][split_name] = len(
                [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS]
            )
    return summary


def print_split_summary(summary):
    print("\nSplit Summary")
    print("Class     | Train | Val  | Test | Total")
    print("----------+-------+------+------|------")
    for class_name in CLASSES:
        tr = summary[class_name]["train"]
        va = summary[class_name]["val"]
        te = summary[class_name]["test"]
        total = tr + va + te
        print(f"{class_name:<9} | {tr:<5} | {va:<4} | {te:<4} | {total}")


def duplicate_leakage_summary(split_dir: Path):
    split_hashes = {split_name: set() for split_name in SPLITS}
    for split_name in SPLITS:
        for class_name in CLASSES:
            for path in (split_dir / split_name / class_name).iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    split_hashes[split_name].add(sha1_of_file(path))

    tv = len(split_hashes["train"] & split_hashes["val"])
    tt = len(split_hashes["train"] & split_hashes["test"])
    vt = len(split_hashes["val"] & split_hashes["test"])
    return tv, tt, vt


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"

    if not processed_dir.is_dir():
        raise SystemExit(f"Missing folder: {processed_dir}")

    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise SystemExit("Split ratios must sum to 1.0")

    images_by_class = {}
    for class_name in CLASSES:
        class_dir = processed_dir / class_name
        images_by_class[class_name] = get_class_images(class_dir)
        if not images_by_class[class_name]:
            raise SystemExit(f"No images found for class: {class_name}")

    targets = build_targets_by_class(images_by_class)
    hash_groups = build_hash_groups(images_by_class)

    split_dir = reset_split_structure(data_dir)
    digest_to_split, _ = assign_groups_to_splits(hash_groups, targets)
    copy_files_to_split(split_dir, hash_groups, digest_to_split)

    summary = summarize_split_counts(split_dir)
    print_split_summary(summary)

    tv, tt, vt = duplicate_leakage_summary(split_dir)
    print("\nDuplicate Content Across Splits (hash overlap)")
    print(f"train-val:  {tv}")
    print(f"train-test: {tt}")
    print(f"val-test:   {vt}")


if __name__ == "__main__":
    main()
