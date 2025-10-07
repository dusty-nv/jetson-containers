#!/usr/bin/env bash
echo $NVCV_BINARIES > urls.txt
sed -i 's|;|\n|g' urls.txt
echo "Downloading files:"
cat urls.txt

downloads="$PWD/downloads"
mkdir $downloads
cd $downloads

set -ex
wget $WGET_FLAGS -i ../urls.txt

# jetson binaries come as a bundle
for filename in *.zip; do
  [ -e "$filename" ] || continue;
  mkdir extracted
  cd extracted
  unzip ../$filename
  ls -R .
  cd ./*
done

# on jetson, this will be inside bundle
if [ -f cvcuda-lib-*.deb ]; then
  dpkg -i cvcuda-lib-*.deb
fi

if [ -f cvcuda-dev-*.deb ]; then
  dpkg -i cvcuda-dev-*.deb
fi

if [ -f cvcuda-test-*.deb ]; then
  dpkg -i cvcuda-test-*.deb
fi

# install python wheels
for filename in *.whl; do
  if [[ -f $filename && "$NVCV_PYTHON" == "on" ]]; then
    uv pip install $filename
  fi
done

# cleanup /tmp
rm -rf $downloads
