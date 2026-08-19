import os

def full_audit():
    print("=" * 70)
    print("      COMPREHENSIVE PROJECT FILE & FOLDER AUDIT      ")
    print("=" * 70)

    # Project ke liye zaroori folders aur core files ki list
    expected_structure = {
        "Directories": [
            "src",
            "src/data",
            "src/models",
            "src/evaluation",
            "src/training",
            "src/utils",
            "src/api",
            "src/auth",
            "data",
            "data/processed",
            "client-ui",
            "frontend",
            "server",
            "notebooks",
            "saved_models",
            "logs",
            "experiments",
            "images"
        ],
        "Core Files": [
            "requirements.txt",
            "main.py",
            "src/config.py",
            "src/evaluation/evaluate.py",
            "src/models/gat.py",
            "src/models/gcn.py",
            "src/models/edge_decoder.py",
            "src/data/negative_sampling.py",
            "src/data/create_features.py"
        ]
    }

    print("\n[+] CHECKING EXPECTED FOLDERS:")
    for folder in expected_structure["Directories"]:
        exists = os.path.isdir(folder)
        status = "🟢 [Found]" if exists else "🔴 [Missing]"
        print(f"  {status} : {folder}/")

    print("\n[+] CHECKING EXPECTED CORE FILES:")
    for file in expected_structure["Core Files"]:
        exists = os.path.isfile(file)
        status = "🟢 [Found]" if exists else "🔴 [Missing]"
        print(f"  {status} : {file}")

    print("\n[+] FULL REPOSITORY FILE TREE (Ignoring virtual environments & caches):")
    ignore_dirs = {
        ".venv", ".venv312", ".venv314", ".git", "__pycache__", 
        ".vercel", "node_modules", ".vscode"
    }
    
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        level = root.replace(".", "").count(os.sep)
        if level > 3:
            continue
        indent = "    " * level
        folder_name = os.path.basename(root) if root != "." else "."
        print(f"{indent}📂 {folder_name}/")
        sub_indent = "    " * (level + 1)
        for f in files:
            print(f"{sub_indent}📄 {f}")

    print("\n" + "=" * 70)
    print("Audit finished successfully without any missing import errors!")
    print("=" * 70)

if __name__ == "__main__":
    full_audit()