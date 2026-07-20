#!/usr/bin/env bash
set -ex

echo "Installing FFMPEG from $FFMPEG_INSTALL"

if [ "$FFMPEG_INSTALL" == "jetpack" ]; then
  APT_URL="https://repo.download.nvidia.com/jetson"
  VERSION="=*nvidia"
  apt-key adv --fetch-key $APT_URL/jetson-ota-public.asc
  printf "deb $APT_URL/ffmpeg main main\ndeb-src $APT_URL/ffmpeg main main\n" | tee -a /etc/apt/sources.list
  printf "Package: *\nPin: origin \"repo.download.nvidia.com\"\nPin-Priority: 999\n" | tee -a /etc/apt/preferences
fi

if [ "$FFMPEG_INSTALL" == "apt" ] || [ "$FFMPEG_INSTALL" == "jetpack" ]; then
  apt-get purge -y ffmpeg || true
  apt-get update
  apt-get install -y --no-install-recommends \
	    libavcodec-dev \
      libavdevice-dev \
	    libavfilter-dev \
	    libavformat-dev \
	    libavutil-dev \
      libpostproc-dev \
	    libswscale-dev \
      libswresample-dev \
      ffmpeg$VERSION
  apt-get clean
  rm -rf /var/lib/apt/lists/*
elif [ "$FFMPEG_INSTALL" == "git" ]; then
  apt-get update
  apt-get install -y --no-install-recommends \
      libass-dev \
      libfreetype6-dev \
      libgnutls28-dev \
      libmp3lame-dev \
      libsdl2-dev \
      libtool \
      libva-dev \
      libvdpau-dev \
      libvorbis-dev \
      libxcb1-dev \
      libxcb-shm0-dev \
      libxcb-xfixes0-dev \
      libvpx-dev \
      libx264-dev \
      libx265-dev \
      libopus-dev
  # Note: do not install Ubuntu 22.04's libdav1d-dev (0.9.2). FFmpeg 7+/8+
  # requires dav1d >= 1.0.0; build.sh compiles a modern dav1d from source.
  apt-get clean
  rm -rf /var/lib/apt/lists/*

  # Fail the image build if both the tarpack cache and the source build fail.
  # The previous `|| echo FAILED` form masked build failures and produced
  # images without an ffmpeg binary (test then failed with command not found).
  tarpack install "ffmpeg-$FFMPEG_VERSION" || $TMP/build.sh
  ldconfig

  # Sanity check so a silent partial install cannot pass the Docker layer.
  command -v ffmpeg >/dev/null 2>&1 || {
    echo "ERROR: ffmpeg binary not found after install (FFMPEG_VERSION=$FFMPEG_VERSION)"
    exit 1
  }
  ffmpeg -version
else
  echo "FFMPEG_INSTALL should be set to 'apt', 'git', or 'jetpack'  (was $FFMPEG_INSTALL)"
  exit 127
fi
