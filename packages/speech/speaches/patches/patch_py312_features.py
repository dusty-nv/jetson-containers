import os
import re

def patch_py312_features(root_dir="."):
    patched_files = []
    func_generic_pattern = re.compile(
        r'^(\s*)def\s+(\w+)\[([A-Za-z_][A-Za-z0-9_]*):\s*([^\]]+)\]\s*\((.*)\)\s*->\s*(.+):'
    )
    class_generic_pattern = re.compile(r'^(\s*)class\s+(\w+)\[([A-Za-z_][A-Za-z0-9_]*)\]')
    type_alias_start_pattern = re.compile(r'^\s*type\s+(\w+)\s*=\s*(.*)')
    fstring_debug_pattern = re.compile(r'\{([^{}=]+)=\}')
    typing_import_pattern = re.compile(r'^from\s+typing\s+import\s+.*')
    future_import_pattern = re.compile(r'^from\s+__future__\s+import\s+')

    for subdir, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue

            path = os.path.join(subdir, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            modified = False
            new_lines = []
            extra_typevars = []
            insert_after = 0
            found_typing_import = False
            has_typevar_import = False

            for i, line in enumerate(lines):
                original = line

                if typing_import_pattern.match(line):
                    found_typing_import = True
                    if "TypeVar" in line:
                        has_typevar_import = True

                if future_import_pattern.match(line):
                    insert_after = i + 1

                # --- Patch function generics ---
                match = func_generic_pattern.match(line)
                if match:
                    indent, func_name, typevar, bound, args, return_type = match.groups()
                    typevar_decl = f"{indent}{typevar} = TypeVar('{typevar}', bound={bound})\n"
                    extra_typevars.append((i, typevar_decl))
                    line = f"{indent}def {func_name}({args}) -> {return_type}:\n"
                    print(f"[PATCHED func generic] {path}:{i+1}: {original.strip()} → {line.strip()}")
                    modified = True

                # --- Patch class generics ---
                match = class_generic_pattern.match(line)
                if match:
                    indent, class_name, typevar = match.groups()
                    typevar_decl = f"{indent}{typevar} = TypeVar('{typevar}')\n"
                    extra_typevars.append((i, typevar_decl))
                    line = f"{indent}class {class_name}(Generic[{typevar}]):\n"
                    print(f"[PATCHED class generic] {path}:{i+1}: {original.strip()} → {line.strip()}")
                    modified = True

                # --- Patch f"{x=}" to f"{'x=' + str(x)}"
                if "f" in line and "{=" in line:
                    patched_line = fstring_debug_pattern.sub(
                        lambda m: f'{{"{m.group(1)}=" + str({m.group(1)})}}', line
                    )
                    if patched_line != line:
                        print(f"[PATCHED fstring] {path}:{i+1}: {line.strip()} → {patched_line.strip()}")
                        line = patched_line
                        modified = True

                # --- Convert `type Alias = ...` to plain assignment
                match = type_alias_start_pattern.match(line)
                if match:
                    alias, rhs = match.groups()
                    line = f"{alias} = {rhs.strip()}\n"
                    print(f"[PATCHED type alias] {path}:{i+1}: {original.strip()} → {line.strip()}")
                    modified = True

                new_lines.append(line)

            # Insert TypeVar import if needed
            if extra_typevars and not has_typevar_import:
                import_line = "from typing import TypeVar, Generic\n"
                new_lines.insert(insert_after, import_line)
                print(f"[INSERTED import] {path}:{insert_after+1}: {import_line.strip()}")
                modified = True

            # Insert all TypeVar definitions (from bottom to top to preserve offsets)
            for offset, def_line in reversed(extra_typevars):
                new_lines.insert(offset, def_line)

            if modified:
                backup_path = path + ".bak_py312"
                os.rename(path, backup_path)
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                patched_files.append(path)

    print(f"\n✅ Patched {len(patched_files)} file(s) for Python 3.12+ compatibility.\n")

if __name__ == "__main__":
    patch_py312_features("/opt/speaches/src")  # Update path as needed
