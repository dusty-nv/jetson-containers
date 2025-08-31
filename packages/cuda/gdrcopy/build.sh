#!/usr/bin/env bash
set -euxo pipefail

echo "Installing NVIDIA GDRCopy ${GDRCOPY_VERSION:-unknown} (GDRCopy)"

# 1) Build .deb packages
git clone --recursive https://github.com/NVIDIA/gdrcopy.git /opt/gdrcopy
cd /opt/gdrcopy/packages

# Drop dkms dependency from the meta package to avoid kernel module build
sed -i 's/gdrdrv-dkms (= @FULL_VERSION@), //g' debian-meta/control

CUDA=/usr/local/cuda ./build-deb-packages.sh

mkdir -p debs
mv ./*.deb debs || true

# 2) Remove the dkms driver package if it was produced
rm -f debs/gdrdrv-dkms*.deb || true

echo "Built packages:"
ls -la debs

EXTRACT_DIR="$(pwd)/build/pkg/txz/lib"
mkdir -p "${EXTRACT_DIR}"

# Extract each .deb payload into the lib folder (dpkg-deb preserves paths under /usr)
for pkg in debs/*.deb; do
  dpkg-deb -x "$pkg" "${EXTRACT_DIR}"
done

dpkg -i debs/*.deb || apt-get -f install -y && dpkg -i debs/*.deb

ls -ld "${EXTRACT_DIR}" || true
ls -la "${EXTRACT_DIR}" || true
ldconfig -p | grep -i gdrapi || true

shopt -s nullglob
for f in build/pkg/txz/*.txz; do
  tar -xvJf "$f" -C build/pkg/txz/lib --strip-components=1
  tar -xvJf "$f" -C /usr/local/ --strip-components=1
done
shopt -u nullglob
tarpack upload "gdrcopy-${GDRCOPY_VERSION}" "${EXTRACT_DIR}" || echo "failed to upload tarball"

# 7) Cleanup
cd /
rm -rf /opt/gdrcopy
ldconfig
