#!/usr/bin/env bash
set -ex

echo "Installing FFMPEG from $FFMPEG_INSTALL"

if [ "$FFMPEG_INSTALL" == "jetpack" ]; then
  echo "deb https://repo.download.nvidia.com/jetson/ffmpeg main main" | tee -a /etc/apt/sources.list
  echo "deb-src https://repo.download.nvidia.com/jetson/ffmpeg main main" | tee -a /etc/apt/sources.list
fi

if [ "$FFMPEG_INSTALL" == "apt" ] || [ "$FFMPEG_INSTALL" == "jetpack" ]; then
  apt-get purge -y ffmpeg || true
  apt-get update
  apt-get install -y --no-install-recommends \
	    ffmpeg \
	    libavcodec-dev \
	    libavfilter-dev \
	    libavformat-dev \
	    libavutil-dev \
	    libavdevice-dev
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
      libopus-dev \
      libdav1d-dev
  apt-get clean
  rm -rf /var/lib/apt/lists/*

  tarpack install "ffmpeg-$FFMPEG_VERSION" || \
  $TMP/build.sh || \
  echo "FAILED to build FFMPEG $FFMPEG_VERSION"
  ldconfig
else
  echo "FFMPEG_INSTALL should be set to 'apt', 'git', or 'jetpack'  (was $FFMPEG_INSTALL)"
  exit 127
fi
