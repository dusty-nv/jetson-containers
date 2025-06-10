#---
# name: pl2303
# group: hardware
# config: config.py
# depends: [build-essential]
# test: load.sh
# notes: Kernel module builder for PL2303 serial-to-USB converter
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG KERNEL_URL
COPY Module.symvers /tmp
SHELL ["/bin/bash", "-c"] 
WORKDIR /usr/src

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  flex \
		  bison \
          libssl-dev \
          bc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN wget $WGET_FLAGS $KERNEL_URL && \
    tar -xvf public_sources.tbz2 Linux_for_Tegra/source/kernel_src.tbz2 --strip-components=2 && \
    tar -xvf kernel_src.tbz2 && \
    rm *.tbz2

#sed -i 's/# CONFIG_USB_SERIAL_FTDI_SIO is not set/CONFIG_USB_SERIAL_FTDI_SIO=m/' .config
#sudo sed -i 's/# CONFIG_USB_SERIAL_CP210X is not set/CONFIG_USB_SERIAL_CP210X=m/' .config
RUN source kernel_src_build_env.sh && \
    cd kernel/$KERNEL_SRC_DIR && \
    make defconfig && \
    sed -i 's|CONFIG_LOCALVERSION=""|CONFIG_LOCALVERSION="-tegra"|' .config && \
    sed -i 's|.*CONFIG_USB_SERIAL_PL2303.*|CONFIG_USB_SERIAL_PL2303=m|' .config && \
    grep 'CONFIG_LOCALVERSION' .config && \
    grep 'CONFIG_USB_SERIAL' .config && \
    grep 'CONFIG_USB_SERIAL_PL2303' .config && \
    cp /tmp/Module.symvers . && \
    make modules_prepare && \
    make M=drivers/usb/serial/