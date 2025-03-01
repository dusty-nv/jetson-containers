#!/usr/bin/env python3
import difflib
import os
import re

def modify_CMakeLists(content):
    lines = content.splitlines()
    new_lines = []
    for line in lines:
        # Reemplazar la definición de CUDA_SUPPORTED_ARCHS
        if re.search(r'^set\s*\(\s*CUDA_SUPPORTED_ARCHS\s+["\'].*["\']\s*\)', line):
            new_lines.append('set(CUDA_SUPPORTED_ARCHS "8.7")')
            continue

        # Para cada llamada a cuda_archs_loose_intersection, reemplazar el segundo argumento por "${CUDA_ARCHS}"
        if "cuda_archs_loose_intersection(" in line:
            # Se espera un formato: cuda_archs_loose_intersection(<NAME> "<ARG2>" "<ARG3>")
            # Queremos reemplazar <ARG2> por "${CUDA_ARCHS}" sin modificar <NAME> ni <ARG3>.
            pattern = r'(cuda_archs_loose_intersection\(\S+\s+)"[^"]+"\s+("[^"]+"\))'
            repl = r'\1"${CUDA_ARCHS}" \2'
            newline = re.sub(pattern, repl, line)
            new_lines.append(newline)
            continue

        new_lines.append(line)
    return "\n".join(new_lines) + "\n"

def modify_vllm_flash_attn_cmake(content):
    lines = content.splitlines()
    new_lines = []
    inserted_patch = False
    for i, line in enumerate(lines):
        # Antes de la condición if(VLLM_FLASH_ATTN_SRC_DIR), insertar la definición de patch_vllm_flash_attn
        if not inserted_patch and re.search(r'^\s*if\s*\(DEFINED\s+ENV\{VLLM_FLASH_ATTN_SRC_DIR\}\)', line):
            new_lines.append('set(patch_vllm_flash_attn git apply /tmp/vllm/fa.diff)')
            inserted_patch = True
        new_lines.append(line)
        # Dentro de cada FetchContent_Declare, después de la línea que define BINARY_DIR, insertar PATCH_COMMAND y UPDATE_DISCONNECTED
        if re.search(r'^\s*BINARY_DIR\s+\$\{CMAKE_BINARY_DIR\}/vllm-flash-attn', line):
            new_lines.append('          PATCH_COMMAND ${patch_vllm_flash_attn}')
            new_lines.append('          UPDATE_DISCONNECTED 1')
    return "\n".join(new_lines) + "\n"

def modify_guided_decoding_init(content):
    # En este archivo se desea eliminar un bloque específico
    # Buscamos la línea de inicio (comentario) y la línea de finalización (comentario de fallback)
    lines = content.splitlines()
    new_lines = []
    skip = False
    for line in lines:
        if not skip and "xgrammar only has x86 wheels for linux" in line:
            skip = True
            continue  # omitir esta línea
        if skip and "xgrammar doesn't support regex" in line:
            skip = False
            new_lines.append(line)  # conservar la línea de fallback
            continue
        if not skip:
            new_lines.append(line)
    return "\n".join(new_lines) + "\n"

def generate_diff_for_file(original_file, modify_function, output_diff):
    with open(original_file, "r") as f:
        original_content = f.read()
    modified_content = modify_function(original_content)
    # Guardar el archivo modificado (por ejemplo, en un archivo temporal)
    mod_file = original_file + ".modified"
    with open(mod_file, "w") as f:
        f.write(modified_content)
    diff = difflib.unified_diff(
        original_content.splitlines(),
        modified_content.splitlines(),
        fromfile="a/" + os.path.basename(original_file),
        tofile="b/" + os.path.basename(original_file),
        lineterm=""
    )
    with open(output_diff, "w") as f:
        f.write("\n".join(diff) + "\n")
    print(f"Diff for {original_file} written to {output_diff}")

def main():
    # Define las rutas de los archivos a modificar
    base_dir = "/opt/vllm"  # ajusta según corresponda
    files_to_modify = {
        os.path.join(base_dir, "CMakeLists.txt"): modify_CMakeLists,
        os.path.join(base_dir, "cmake", "external_projects", "vllm_flash_attn.cmake"): modify_vllm_flash_attn_cmake,
        os.path.join(base_dir, "vllm", "model_executor", "guided_decoding", "__init__.py"): modify_guided_decoding_init,
    }
    # Directorio donde se almacenarán los diffs
    diff_dir = "/tmp/vllm"
    os.makedirs(diff_dir, exist_ok=True)
    for filepath, mod_func in files_to_modify.items():
        diff_file = os.path.join(diff_dir, os.path.basename(filepath) + ".diff")
        generate_diff_for_file(filepath, mod_func, diff_file)

if __name__ == "__main__":
    main()
