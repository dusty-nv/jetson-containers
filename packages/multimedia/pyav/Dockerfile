#---
# name: pyav
# group: multimedia
# depends: [python, cmake]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FFMPEG_VERSION=7.0.2

# Building FFmpeg for libav*** libraries, which are necessory for PyAV
RUN cd /opt && \
    wget https://www.ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.gz && \
    tar -xvzf ffmpeg* && \
    rm ffmpeg*.tar.gz

# libaom for AV1
RUN cd /opt/ffmpeg*  && \
    mkdir -p ./libaom && \
    cd ./libaom && \
    git clone https://aomedia.googlesource.com/aom && \
    mkdir -p aom_build && \
    cd aom_build && \
    PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DENABLE_TESTS=OFF -DENABLE_NASM=on ../aom && \
    PATH="$HOME/bin:$PATH" make && \
    make install

# libsvtav1 for AV1
RUN cd /opt/ffmpeg*  && \
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
RUN cd /opt/ffmpeg*  && \
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
ARG PKG_CONFIG_PATH=/opt/ffmpeg-${FFMPEG_VERSION}/libavdevice/:/opt/ffmpeg-${FFMPEG_VERSION}/libavfilter/:/opt/ffmpeg-${FFMPEG_VERSION}/libavformat/:/opt/ffmpeg-${FFMPEG_VERSION}/libavcodec/:/opt/ffmpeg-${FFMPEG_VERSION}/libswresample/:/opt/ffmpeg-${FFMPEG_VERSION}/libswscale/:/opt/ffmpeg-${FFMPEG_VERSION}/libavutil/
RUN pip3 install --no-cache-dir --verbose pyav
