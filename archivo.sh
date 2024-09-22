#!/bin/bash

# Directorio raíz desde donde empezar a buscar
ROOT_DIR=${1:-.}

# Encontrar todos los archivos .sh y cambiar permisos
find "$ROOT_DIR" -type f -name "*.sh" -exec chmod u+x {} \;

echo "Permisos cambiados para todos los archivos .sh en $ROOT_DIR"

