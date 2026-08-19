import os

# Output file ka naam jisme saara code save hoga
output_file = "all_code_combined.txt"

# Kin extensions wali files ko include karna hai
valid_extensions = (".py", ".json", ".md", ".tsx", ".ts", ".js", ".env")

# Kin folders ko ignore karna hai (jaise virtual environment, node_modules, .git)
ignored_dirs = {".venv", ".venv312", ".venv_py314", "node_modules", ".git", ".next"}

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk("."):
        # Ignored directories ko skip karna
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            if file.endswith(valid_extensions) and file != output_file:
                file_path = os.path.join(root, file)
                
                # File header likhein taaki pata chale kaun sa code kahan se hai
                outfile.write("=" * 80 + "\n")
                outfile.write(f"FILE: {file_path}\n")
                outfile.write("=" * 80 + "\n\n")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"[Error reading file: {e}]\n")
                
                outfile.write("\n\n" + "-" * 80 + "\n\n")

print(f"Success! Saara code '{output_file}' mein save ho gaya hai.")