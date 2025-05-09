import os
import json

def scan_directory(root_dir):
    """
    Scans a directory and returns its structure in JSON format,
    skipping JPG, JPEG, and TXT files.
    """
    directory_structure = {
        "name": os.path.basename(root_dir),
        "type": "directory",
        "children": []
    }

    skip_extensions = {'.jpg', '.jpeg', '.txt'}

    try:
        for entry in os.listdir(root_dir):
            full_path = os.path.join(root_dir, entry)
            
            if os.path.isdir(full_path):
                # Recursively scan subdirectories
                directory_structure["children"].append(scan_directory(full_path))
            else:
                # Skip specified file extensions
                ext = os.path.splitext(entry)[1].lower()
                if ext not in skip_extensions:
                    directory_structure["children"].append({
                        "name": entry,
                        "type": "file"
                    })
    except PermissionError:
        print(f"Permission denied: {full_path}")

    return directory_structure

def save_to_json(data, output_file="directory_structure.json"):
    """Saves the directory structure to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Directory structure saved to {output_file}")

if __name__ == "__main__":
    target_dir = input("Enter directory to scan (press Enter for current): ") or "."
    structure = scan_directory(os.path.abspath(target_dir))
    save_to_json(structure)