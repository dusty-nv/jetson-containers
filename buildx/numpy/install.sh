set -ex

pip3 install --force-reinstall ${NUMPY_PACKAGE}
pip3 show numpy && python3 -c 'import numpy; print(numpy.__version__)'

set +e

# some libraries are more tightly-coupled with numpy and might not like minor version changes.
# for example, numba 0.61 requires numpy<2.2.  
pip3 show numba

if [ $? = 0 ]; then
  python3 -c 'import numba'
  if [ $? != 0 ]; then # numba failed to import (presumably due to numpy being changed)
    pip3 install --force-reinstall numba 
  fi
fi