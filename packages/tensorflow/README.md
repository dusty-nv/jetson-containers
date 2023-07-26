# tensorflow

<details open>
<summary><h3>tensorflow</h3></summary>

|            |            |
|------------|------------|
| Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`protobuf:cpp`](/packages/protobuf/protobuf_cpp) |
| Dependants | [`l4t-tensorflow:tf1`](/packages/l4t/l4t-tensorflow) |
| Dockerfile | [`Dockerfile`](Dockerfile) |

The [`l4t-tensorflow`](/packages/l4t/l4t-tensorflow) containers are similar, with the addition of OpenCV:CUDA and PyCUDA.  

The TensorFlow wheels used in these are from https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform

</details>
<details open>
<summary><h3>tensorflow2</h3></summary>

|            |            |
|------------|------------|
| Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`protobuf:cpp`](/packages/protobuf/protobuf_cpp) |
| Dependants | [`l4t-ml`](/packages/l4t/l4t-ml) [`l4t-tensorflow:tf2`](/packages/l4t/l4t-tensorflow) |
| Dockerfile | [`Dockerfile`](Dockerfile) |
</details>
