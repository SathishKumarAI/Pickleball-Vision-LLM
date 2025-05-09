import os

# Specify the root directory of your codebase
root_dir = "/workspaces/Pickleball-Vision-LLM"

# Walk through the directory structure
for dirpath, dirnames, filenames in os.walk(root_dir):
    # Skip hidden directories and files
    dirnames[:] = [d for d in dirnames if not d.startswith('.')]
    filenames = [f for f in filenames if not f.startswith('.')]
    
    print(f"Directory: {dirpath}")
    for dirname in dirnames:
        print(f"  Subdirectory: {dirname}")
    for filename in filenames:
        print(f"  File: {filename}")
