import re
from pathlib import Path

def extract_trailing_number(filename: str) -> str | None:
    """Extracts the last group of digits from the filename (before the extension)."""
    matches = re.findall(r"\d+", filename)
    return matches[-1] if matches else None

def rename_files_to_trailing_number(folder_path: str) -> None:
    """
    Recursively renames files in the folder to just the trailing number in their filename.
    Skips files with no number in the name.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder_path} is not a valid directory")

    for file_path in folder.rglob("*"):
        if file_path.is_file():
            number = extract_trailing_number(file_path.stem)
            if number:
                new_name = f"{number}{file_path.suffix}"
                new_path = file_path.with_name(new_name)

                # Handle naming conflicts
                count = 1
                while new_path.exists():
                    new_name = f"{number}_{count}{file_path.suffix}"
                    new_path = file_path.with_name(new_name)
                    count += 1

                print(f"Renaming: {file_path.name} -- {new_path.name}")
                file_path.rename(new_path)
            else:
                print(f"Skipping: {file_path.name} (no number found)")

# Example Usage
if __name__ == "__main__":
    rename_files_to_trailing_number(r"src/data_collection/src/data_collection/data")
