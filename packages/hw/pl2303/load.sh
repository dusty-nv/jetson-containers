#!/usr/bin/env bash
set -e

MODNAME="pl2303"

if [ -f /.dockerenv ]; then
    source /usr/src/kernel_src_build_env.sh
    set -x
    cp /usr/src/kernel/$KERNEL_SRC_DIR/drivers/usb/serial/$MODNAME.ko .
    ls -ll $MODNAME.ko
else    
    MODSRC="$(dirname "$(readlink -f "$0")")/$MODNAME.ko"
    MODEST="/lib/modules/$(uname -r)"
    echo "Loading $MODNAME.ko kernel module (requires sudo)"
    set -x
    sudo cp $MODSRC $MODEST
    sudo depmod
    sudo modprobe $MODNAME
    sudo lsmod | grep $MODNAME
fi