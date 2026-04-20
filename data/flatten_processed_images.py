from pathlib import Path
import shutil

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CATEGORIES = ("bacterial", "fungal", "stress", "viral")


def unique_target(path: Path) -> Path:
    candidate = path
    i = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}__{i}{path.suffix}")
        i += 1
    return candidate


def remove_empty_tree(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False

    # Remove empty child directories first (deepest to shallowest).
    child_dirs = sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for d in child_dirs:
        try:
            d.rmdir()
        except OSError:
            pass

    try:
        root.rmdir()
        return True
    except OSError:
        return False


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    processed_dir = base_dir / "processed"

    if not processed_dir.is_dir():
        raise SystemExit(f"Missing required folder: {processed_dir}")

    moved_counts = {c: 0 for c in CATEGORIES}
    deleted_subfolders = []
    skipped_files = []

    for category in CATEGORIES:
        category_dir = processed_dir / category
        if not category_dir.is_dir():
            skipped_files.append(f"[missing category] {category_dir.relative_to(base_dir)}")
            continue

        subfolders = sorted(
            (p for p in category_dir.iterdir() if p.is_dir()),
            key=lambda p: p.name.lower(),
        )

        for subfolder in subfolders:
            files = sorted(
                (p for p in subfolder.rglob("*") if p.is_file()),
                key=lambda p: p.relative_to(subfolder).as_posix().lower(),
            )

            for src in files:
                if src.suffix.lower() not in IMAGE_EXTENSIONS:
                    skipped_files.append(
                        f"{src.relative_to(base_dir)} (unsupported extension)"
                    )
                    continue

                target = category_dir / f"{subfolder.name}__{src.name}"
                target = unique_target(target)

                shutil.move(str(src), str(target))
                moved_counts[category] += 1

            if remove_empty_tree(subfolder):
                deleted_subfolders.append(str(subfolder.relative_to(base_dir)))

    print("\n=== Summary ===")
    print(f"Files moved into bacterial: {moved_counts['bacterial']}")
    print(f"Files moved into fungal: {moved_counts['fungal']}")
    print(f"Files moved into stress: {moved_counts['stress']}")
    print(f"Files moved into viral: {moved_counts['viral']}")

    print("\nDeleted subfolders:")
    if deleted_subfolders:
        for folder in deleted_subfolders:
            print(f"- {folder}")
    else:
        print("- None")

    print("\nSkipped files:")
    if skipped_files:
        for item in skipped_files:
            print(f"- {item}")
    else:
        print("- None")


if __name__ == "__main__":
    main()
