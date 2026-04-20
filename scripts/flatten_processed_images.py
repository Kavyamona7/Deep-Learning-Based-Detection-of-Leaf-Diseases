from pathlib import Path
import shutil

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASSES = ("bacterial", "fungal")


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

    child_dirs = sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for directory in child_dirs:
        try:
            directory.rmdir()
        except OSError:
            pass

    try:
        root.rmdir()
        return True
    except OSError:
        return False


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    if not processed_dir.is_dir():
        raise SystemExit(f"Missing required folder: {processed_dir}")

    moved_counts = {class_name: 0 for class_name in CLASSES}
    deleted_subfolders = []
    skipped_files = []

    for class_name in CLASSES:
        class_dir = processed_dir / class_name
        if not class_dir.is_dir():
            skipped_files.append(f"[missing class] {class_dir.relative_to(project_root)}")
            continue

        subfolders = sorted(
            (p for p in class_dir.iterdir() if p.is_dir()),
            key=lambda p: p.name.lower(),
        )

        for subfolder in subfolders:
            files = sorted(
                (p for p in subfolder.rglob("*") if p.is_file()),
                key=lambda p: p.relative_to(subfolder).as_posix().lower(),
            )

            for source_file in files:
                if source_file.suffix.lower() not in IMAGE_EXTENSIONS:
                    skipped_files.append(
                        f"{source_file.relative_to(project_root)} (unsupported extension)"
                    )
                    continue

                target = class_dir / f"{subfolder.name}__{source_file.name}"
                target = unique_target(target)
                shutil.move(str(source_file), str(target))
                moved_counts[class_name] += 1

            if remove_empty_tree(subfolder):
                deleted_subfolders.append(str(subfolder.relative_to(project_root)))

    print("\n=== Flatten Summary ===")
    print(f"Files moved into bacterial: {moved_counts['bacterial']}")
    print(f"Files moved into fungal: {moved_counts['fungal']}")

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
