#!/usr/bin/env bash

set -e
source scripts/docker_base.sh

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TEST_MOUNT="$ROOT/../test:/test"

test_cuda()
{
	echo "testing container $1 => CUDA"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r test/test_cuda.sh
	echo -e "done testing container $1 => CUDA\n"
}

test_tensorrt()
{
	echo "testing container $1 => TensorRT"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_tensorrt.py
	echo -e "done testing container $1 => TensorRT\n"
}

test_opencv()
{
	echo "testing container $1 => OpenCV"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_opencv.py
	echo -e "done testing container $1 => OpenCV\n"
}

test_vpi()
{
	echo "testing container $1 => VPI"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_vpi.py
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r test/test_vpi.sh
	echo -e "done testing container $1 => VPI\n"
}

container_tag="jetpack:r$L4T_VERSION"

test_cuda $container_tag
test_tensorrt $container_tag
test_opencv $container_tag
test_vpi $container_tag