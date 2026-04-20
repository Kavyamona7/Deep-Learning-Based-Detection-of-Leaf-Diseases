from pathlib import Path
import shutil
import random

CLASSES = ("bacterial", "fungal", "stress", "viral")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def ensure_split_structure(base_dir: Path) -> Path:
    split_dir = base_dir / "split"
    for split_name in ("train", "val", "test"):
        for class_name in CLASSES:
            (split_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)
    return split_dir


def get_class_images(class_dir: Path):
    if not class_dir.is_dir():
        return []
    return sorted(
        [
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )


def print_summary(summary):
    headers = ("Class", "Train", "Val", "Test", "Total")
    rows = []
    for class_name in CLASSES:
        counts = summary[class_name]
        rows.append(
            (
                class_name,
                str(counts["train"]),
                str(counts["val"]),
                str(counts["test"]),
                str(counts["total"]),
            )
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)

    print("\nSplit Summary")
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))


def main():
    base_dir = Path(__file__).resolve().parent
    processed_dir = base_dir / "processed"

    if not processed_dir.is_dir():
        raise SystemExit(f"Missing folder: {processed_dir}")

    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise SystemExit("Split ratios must sum to 1.0")

    split_dir = ensure_split_structure(base_dir)
    summary = {}

    for idx, class_name in enumerate(CLASSES):
        class_dir = processed_dir / class_name
        images = get_class_images(class_dir)

        rng = random.Random(SEED + idx)
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        for src in train_files:
            shutil.copy2(src, split_dir / "train" / class_name / src.name)
        for src in val_files:
            shutil.copy2(src, split_dir / "val" / class_name / src.name)
        for src in test_files:
            shutil.copy2(src, split_dir / "test" / class_name / src.name)

        summary[class_name] = {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
            "total": n,
        }

    print_summary(summary)


if __name__ == "__main__":
    main()
