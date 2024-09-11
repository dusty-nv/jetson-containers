#---
# name: lerobot
# group: embodiment
# docs: docs.md
# depends: [torchvision, opencv, huggingface_hub, h5py]
# requires: '>=36'
# test: 
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt

## Building FFmpeg for libav*** libraries, which are necessory for PyAV

# Download FFmpeg
RUN set -ex && \
#    wget https://www.ffmpeg.org/releases/ffmpeg-4.2.10.tar.gz && \
#    wget https://www.ffmpeg.org/releases/ffmpeg-4.3.8.tar.gz && \
#    wget https://www.ffmpeg.org/releases/ffmpeg-4.4.5.tar.gz && \
#    wget https://www.ffmpeg.org/releases/ffmpeg-5.1.6.tar.gz && \
#    wget https://www.ffmpeg.org/releases/ffmpeg-6.1.2.tar.gz && \
    wget https://www.ffmpeg.org/releases/ffmpeg-7.0.2.tar.gz && \
    tar -xvzf ffmpeg*
RUN rm ffmpeg*.tar.gz

# libaom for AV1
RUN cd ffmpeg*  && \
    mkdir -p ./libaom && \
    cd ./libaom && \
    git clone https://aomedia.googlesource.com/aom && \
    mkdir -p aom_build && \
    cd aom_build && \
    PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DENABLE_TESTS=OFF -DENABLE_NASM=on ../aom && \
    PATH="$HOME/bin:$PATH" make && \
    make install

# libsvtav1 for AV1
RUN cd ffmpeg*  && \
    git -C SVT-AV1 pull 2> /dev/null || git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
    mkdir -p SVT-AV1/build && \
    cd SVT-AV1/build && \
    PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF .. && \
    PATH="$HOME/bin:$PATH" make && \
    make install

# libdav1d for AV1
RUN set -ex \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends libdav1d-dev \
	&& apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
# Build FFmpeg
RUN cd ffmpeg*  && \
    PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
        --prefix="$HOME/ffmpeg_build" \
        # --pkg-config-flags="--static" \
        --extra-cflags="-I$HOME/ffmpeg_build/include" \
        --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
        --extra-libs="-lpthread -lm" \
        --ld="g++" \
        --bindir="$HOME/bin" \
        --enable-shared --disable-doc \
        --enable-libaom --enable-libsvtav1 --enable-libdav1d && \
    make && \
    make install && \
    ldconfig
ENV LD_LIBRARY_PATH=/root/ffmpeg_build/lib:$LD_LIBRARY_PATH
RUN $HOME/bin/ffmpeg -version
RUN $HOME/bin/ffmpeg -decoders && \
    $HOME/bin/ffmpeg -decoders | grep av1 || true

ARG LEROBOT_REPO=huggingface/lerobot/
ARG LEROBOT_BRANCH=main

RUN echo $PKG_CONFIG_PATH
RUN ls -lR /opt/ffmpeg-7.0.2
ARG PKG_CONFIG_PATH=/opt/ffmpeg-7.0.2/libavdevice/:/opt/ffmpeg-7.0.2/libavfilter/:/opt/ffmpeg-7.0.2/libavformat/:/opt/ffmpeg-7.0.2/libavcodec/:/opt/ffmpeg-7.0.2/libswresample/:/opt/ffmpeg-7.0.2/libswscale/:/opt/ffmpeg-7.0.2/libavutil/

RUN git clone --branch=${LEROBOT_BRANCH} --depth=1 https://github.com/${LEROBOT_REPO} lerobot && \
    cd lerobot && \
    sed 's|^python.*||' -i pyproject.toml && \
    sed 's|^torch.*||' -i pyproject.toml && \
    sed 's|^opencv-python.*||' -i pyproject.toml && \
    sed 's|^torchvision.*||' -i pyproject.toml && \
    sed 's|^h5py.*||' -i pyproject.toml && \
    sed 's|^huggingface-hub.*||' -i pyproject.toml && \
    echo "########## pyproject.toml ###########" && \
    cat -n pyproject.toml

RUN cd lerobot && \
    pip install --ignore-installed -e ".[aloha, pusht]"

# Install the original FFmpeg
RUN set -ex \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends ffmpeg \
	&& apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Jetson-custom FFmpeg 
# COPY nvidia-l4t-apt-source.list .
# RUN mv nvidia-l4t-apt-source.list /etc/apt/sources.list.d/ && \
#     apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
#     apt update
# RUN apt-get install -y --no-install-recommends ffmpeg && \
# 	  apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Test ffmpeg version
RUN set -ex \
	&& ffmpeg -version \
	&& ffmpeg -encoders \
	&& ffmpeg -decoders \
	&& ffmpeg -decoders | grep nvv4l2 || true

# Set the LD_LIBRARY_PATH environment variable
ENV LD_LIBRARY_PATH=/root/ffmpeg_build/lib:$LD_LIBRARY_PATH

WORKDIR /opt/lerobot/
