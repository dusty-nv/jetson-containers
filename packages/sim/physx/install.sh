#!/usr/bin/env bash
set -ex

: "${CUDA_ARCHS:=87;90;100;103;110;120;121}"

git clone --recursive --depth=1 https://github.com/NVIDIA-Omniverse/PhysX /opt/PhysX
cd /opt/PhysX/physx
awk '

  BEGIN{replaced=0}
  /cuCtxCreate\(&mCtx,.*flags.*mDevHandle\);/ && !replaced{
    print "#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000"
    print "    CUctxCreateParams params{};"
    print "    params.flags   = static_cast<unsigned int>(flags);"
    print "    params.device  = mDevHandle;"
    print "    params.type    = CU_CTX_PRIMARY;"
    print "    CUresult status = cuCtxCreate(&mCtx, &params, 0 /*reserved*/, mDevHandle);"
    print "#else"
    print "    CUresult status = cuCtxCreate(&mCtx, static_cast<unsigned int>(flags), mDevHandle);"
    print "#endif"
    replaced=1; next
  }
  {print}
' source/cudamanager/src/CudaContextManager.cpp > source/cudamanager/src/CudaContextManager.cpp.patched
mv source/cudamanager/src/CudaContextManager.cpp.patched source/cudamanager/src/CudaContextManager.cpp

# --- Remove legacy SM70/72 from any PhysX GPU cmake that hard-codes them ---
# (CUDA 13+ dropped prebuilt SASS for these, so nvcc rejects -gencode compute_70)
if nvcc --version | grep -q 'release 13\.'; then
  # Strip cleanly from any cmake/gpu list forms
  find . -type f \( -name "*.cmake" -o -name "CMakeLists.txt" \) -print0 | xargs -0 sed -i \
    -e 's/\(compute_70,sm_70\)\(;*\|"\|'\''\| \)//g' \
    -e 's/\(compute_72,sm_72\)\(;*\|"\|'\''\| \)//g' \
    -e 's/\<70\>;//g' \
    -e 's/;70\>//g' \
    -e 's/\<72\>;//g' \
    -e 's/;72\>//g'
fi

# Generate build trees (clang aarch64 preset)
bash generate_projects.sh linux-aarch64-clang

# Reconfigure existing build dirs with your arch list and relax Clang’s error
for cfg in checked profile release; do
  bd="/opt/PhysX/physx/compiler/linux-aarch64-clang-${cfg}"
  if [[ -f "${bd}/CMakeCache.txt" ]]; then
    echo "Reconfiguring ${bd} (CUDA_ARCHS=${CUDA_ARCHS})"
    (
      cd "${bd}"
      cmake \
        -DPX_GENERATE_GPU_PROJECTS=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_CXX_FLAGS="-Wno-error=missing-include-dirs" \
        -DCMAKE_C_FLAGS="-Wno-error=missing-include-dirs" \
        -DPX_WARNINGS_AS_ERRORS=OFF \
        .
    )
  fi
done

# Build + install release only
cd /opt/PhysX/physx/compiler/linux-aarch64-clang-release
make -j"$(nproc)"
make install
