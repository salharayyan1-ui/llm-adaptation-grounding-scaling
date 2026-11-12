import os
import sys
import ast
import py_compile

def check_file_syntax(filepath: str) -> bool:
    """Check Python syntax using both ast.parse and py_compile."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source, filename=filepath)
        py_compile.compile(filepath, doraise=True)
        return True
    except Exception as e:
        print(f"[SYNTAX ERROR] in {filepath}: {e}")
        return False

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Checking syntax for all Python files in: {root_dir}")
    
    total = 0
    passed = 0
    failed_files = []
    
    for dirpath, _, filenames in os.walk(root_dir):
        # Skip virtualenvs or cache directories
        if any(part.startswith(".") or part in ["__pycache__", "build", "dist"] for part in dirpath.split(os.sep)):
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                total += 1
                full_path = os.path.join(dirpath, fn)
                rel_path = os.path.relpath(full_path, root_dir)
                if check_file_syntax(full_path):
                    passed += 1
                    print(f"  [PASS] {rel_path}")
                else:
                    failed_files.append(rel_path)
                    
    print("\n" + "=" * 50)
    print(f"Total Python files checked: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(failed_files)}")
    print("=" * 50)
    
    if failed_files:
        print("Failed files:")
        for ff in failed_files:
            print(f"  - {ff}")
        sys.exit(1)
    else:
        print("ALL PYTHON FILES PASSED SYNTAX CHECK!")
        sys.exit(0)

if __name__ == "__main__":
    main()
