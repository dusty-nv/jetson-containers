import os
import re

def patch_py311_features(root_dir="."):
    patched_files = []

    for subdir, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue

            path = os.path.join(subdir, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            modified = False
            new_lines = []
            class_stack = []

            for i, line in enumerate(lines):
                stripped = line.strip()

                # ---- Patch 'from datetime import UTC'
                if "from datetime import" in line and "UTC" in line:
                    parts = line.strip().replace("from datetime import", "").split(",")
                    parts = [p.strip() for p in parts if p.strip() != "UTC"]
                    if "timezone" not in parts:
                        parts.append("timezone")
                    new_lines.append(f"from datetime import {', '.join(parts)}\n")
                    new_lines.append("UTC = timezone.utc\n")
                    print(f"[PATCHED UTC] {path}:{i+1}: replaced UTC import")
                    modified = True
                    continue

                # ---- Patch 'from typing import ..., Self'
                if "from typing import" in line and "Self" in line:
                    parts = line.strip().replace("from typing import", "").split(",")
                    parts = [p.strip() for p in parts if p.strip() != "Self"]
                    if parts:
                        new_lines.append(f"from typing import {', '.join(parts)}\n")
                        print(f"[PATCHED Self-import] {path}:{i+1}: removed Self")
                    else:
                        print(f"[REMOVED typing line] {path}:{i+1}: removed entire import line")
                    modified = True
                    continue

                # ---- Track class scopes to support replacing -> Self with -> "ClassName"
                class_match = re.match(r'^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\(|:)', line)
                if class_match:
                    current_indent = len(line) - len(line.lstrip())
                    class_stack.append((current_indent, class_match.group(1)))

                # ---- Exit class scope
                while class_stack:
                    indent, _ = class_stack[-1]
                    if len(line) - len(line.lstrip()) <= indent and not stripped.startswith("class "):
                        class_stack.pop()
                    else:
                        break

                # ---- Patch return type '-> Self' only if it's standalone
                self_return_match = re.search(r'def\s+\w+\s*\(.*\)\s*->\s*Self\s*:', line)
                if self_return_match and class_stack:
                    _, class_name = class_stack[-1]
                    patched_line = line.replace("-> Self", f'-> "{class_name}"')
                    new_lines.append(patched_line)
                    print(f"[PATCHED Self return] {path}:{i+1}: → \"{class_name}\"")
                    modified = True
                    continue

                # --- Detect and comment out incomplete assignments like: `Part =`
                incomplete_assign_match = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(#.*)?\s*$', line)
                if incomplete_assign_match:
                    var_name = incomplete_assign_match.group(1)
                    comment = f"# [AUTO-PATCHED] Incomplete assignment removed: `{var_name} =`\n"
                    print(f"[REMOVED invalid syntax] {path}:{i+1}: {var_name} =")
                    new_lines.append(comment)
                    modified = True
                    i += 1
                    continue

                new_lines.append(line)

            if modified:
                backup_path = path + ".bak_py311"
                os.rename(path, backup_path)
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                patched_files.append(path)

    print(f"\n✅ Patched {len(patched_files)} file(s) for Python 3.11+ compatibility.\n")

if __name__ == "__main__":
    patch_py311_features("/opt/speaches/src")  # Update to your source path if needed
