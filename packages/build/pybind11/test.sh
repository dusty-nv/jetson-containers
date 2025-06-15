#!/usr/bin/env bash
set -e
printf "Testing pybind11[global]\n"

pip3 show pybind11

ls -R /usr/include/pybind11

python3 -c 'import pybind11; print(f"pybind11 version: {pybind11.__version__}")'

printf "\npybind11 OK\n"