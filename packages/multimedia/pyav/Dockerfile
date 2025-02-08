#---
# name: pyav
# group: multimedia
# depends: [python, cmake]
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FFMPEG_VERSION=7.1

# Building FFmpeg for libav*** libraries, which are necessory for PyAV
RUN cd /opt && \
    wget https://www.ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.gz && \
    tar -xvzf ffmpeg-${FFMPEG_VERSION}.tar.gz && \
    rm ffmpeg-${FFMPEG_VERSION}.tar.gz

# libaom for AV1
RUN cd /opt/ffmpeg-${FFMPEG_VERSION}  && \
    mkdir -p ./libaom && \
    cd ./libaom && \
    git clone https://aomedia.googlesource.com/aom && \
    mkdir -p aom_build && \
    cd aom_build && \
    PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" \
        -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" \
        -DENABLE_TESTS=OFF \
        -DENABLE_NASM=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DAOM_EXTRA_C_FLAGS="-fno-lto" \
        ../aom \
    PATH="$HOME/bin:$PATH" make && \
    make install

# libsvtav1 for AV1
# temporal version 2.3.0, they breaks ffmpeg build: https://gitlab.com/AOMediaCodec/SVT-AV1/-/commit/988e930c1083ce518ead1d364e3a486e9209bf73#900962ec0dfb11881a5f25ce6fcad8e815c8fd45_1056_1122
# solution mid-february: https://gitlab.com/AOMediaCodec/SVT-AV1/-/merge_requests/2355#note_2312506245
RUN cd /opt/ffmpeg-${FFMPEG_VERSION}  && \
    git -C SVT-AV1 pull 2> /dev/null || git clone --recursive https://gitlab.com/AOMediaCodec/SVT-AV1.git && \
    cd SVT-AV1 && \
    git checkout v2.3.0 && \
    cd .. && \
    mkdir -p SVT-AV1/build
RUN cd /opt/ffmpeg-${FFMPEG_VERSION}/SVT-AV1/build && \
    PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" \
        -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" \
        -DCMAKE_BUILD_TYPE=Release \
        # -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF .. && \
    PATH="$HOME/bin:$PATH" make -j$(nproc) && \
    make install

RUN find / -name libSvtAv1Enc.a
RUN ls $HOME/ffmpeg_build/lib/*.a

ENV PKG_CONFIG_PATH="/root/ffmpeg_build/lib/pkgconfig:$PKG_CONFIG_PATH"

# libdav1d for AV1
RUN set -ex \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends libdav1d-dev \
	&& apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
# # Verify FFmpeg directory
# RUN ls -l /opt/ | grep ffmpeg
# RUN ls -l /opt/ffmpeg-${FFMPEG_VERSION}
# RUN find / -name svt-av1.pc

RUN pkg-config --modversion aom
RUN pkg-config --modversion SvtAv1Enc

# Build FFmpeg
RUN cd /opt/ffmpeg-${FFMPEG_VERSION}  && \
    PATH="$HOME/bin:$PATH" ./configure \
        --prefix="$HOME/ffmpeg_build" \
        --extra-cflags="-I$HOME/ffmpeg_build/include -fno-lto" \
        --extra-ldflags="-L$HOME/ffmpeg_build/lib -fno-lto" \
        --extra-libs="-lpthread -lm" \
        --ld="g++" \
        --bindir="$HOME/bin" \
        --enable-shared \
        --disable-doc \
        --enable-libaom \
        --enable-libsvtav1 \
        --enable-libdav1d && \
    make -j$(nproc) && \
    make install && \
    ldconfig
    
ENV PATH="/root/bin:${PATH}"
ENV LD_LIBRARY_PATH=/root/ffmpeg_build/lib:$LD_LIBRARY_PATH
ARG PKG_CONFIG_PATH=/opt/ffmpeg-${FFMPEG_VERSION}/libavdevice/:/opt/ffmpeg-${FFMPEG_VERSION}/libavfilter/:/opt/ffmpeg-${FFMPEG_VERSION}/libavformat/:/opt/ffmpeg-${FFMPEG_VERSION}/libavcodec/:/opt/ffmpeg-${FFMPEG_VERSION}/libswresample/:/opt/ffmpeg-${FFMPEG_VERSION}/libswscale/:/opt/ffmpeg-${FFMPEG_VERSION}/libavutil/
RUN pip3 install --no-cache-dir --verbose pyav
