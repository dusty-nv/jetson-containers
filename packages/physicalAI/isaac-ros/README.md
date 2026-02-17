# isaac-ros

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`isaac-ros:common-3.2-humble-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `isaac-ros:common` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:humble-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) |
| &nbsp;&nbsp;&nbsp;Dependants | [`isaac-ros:compression-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:data-tools-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:manipulator-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nitros-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`isaac-ros:common-3.2-jazzy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:jazzy-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) |
| &nbsp;&nbsp;&nbsp;Dependants | [`isaac-ros:compression-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:data-tools-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:manipulator-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nitros-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |

| **`isaac-ros:nitros-3.2-humble-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `isaac-ros:nitros` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:humble-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-humble-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) |
| &nbsp;&nbsp;&nbsp;Dependants | [`isaac-ros:compression-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:manipulator-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:nitros-3.2-jazzy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:jazzy-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) |
| &nbsp;&nbsp;&nbsp;Dependants | [`isaac-ros:compression-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:manipulator-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:image-pipeline-3.2-humble-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `isaac-ros:image-pipeline` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:humble-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-humble-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dependants | [`isaac-ros:compression-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:image-pipeline-3.2-jazzy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:jazzy-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dependants | [`isaac-ros:compression-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:dnn-inference-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:pose-estimation-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:dnn-inference-3.2-humble-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `isaac-ros:dnn-inference` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:humble-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-humble-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:dnn-inference-3.2-jazzy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:jazzy-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:compression-3.2-humble-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `isaac-ros:compression` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:humble-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-humble-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dependants | [`isaac-ros:pose-estimation-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:compression-3.2-jazzy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:jazzy-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dependants | [`isaac-ros:pose-estimation-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:visual-slam-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:visual-slam-3.2-humble-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `isaac-ros:visual-slam` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:humble-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-humble-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:compression-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:visual-slam-3.2-jazzy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:jazzy-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:compression-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:pose-estimation-3.2-humble-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `isaac-ros:pose-estimation` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:humble-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-humble-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:compression-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:pose-estimation-3.2-jazzy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:jazzy-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:image-pipeline-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:compression-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:nvblox-3.2-humble-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `isaac-ros:nvblox` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:humble-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-humble-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dependants | [`isaac-ros:manipulator-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:nvblox-3.2-jazzy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:jazzy-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dependants | [`isaac-ros:manipulator-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:manipulator-3.2-humble-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `isaac-ros:manipulator` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:humble-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-humble-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-humble-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:manipulator-3.2-jazzy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:jazzy-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`cuda-python`](/packages/cuda/cuda-python) [`cv-cuda:cpp`](/packages/cv/cv-cuda) [`isaac-ros:nitros-3.2-jazzy-desktop`](/packages/robots/isaac-ros) [`isaac-ros:nvblox-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:data-tools-3.2-humble-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Aliases | `isaac-ros:data-tools` |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:humble-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-humble-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

| **`isaac-ros:data-tools-3.2-jazzy-desktop`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`tensorrt`](/packages/cuda/tensorrt) [`numpy`](/packages/numeric/numpy) [`opengl`](/packages/multimedia/opengl) [`cmake`](/packages/build/cmake/cmake_pip) [`llvm`](/packages/build/llvm) [`vulkan`](/packages/multimedia/vulkan) [`video-codec-sdk`](/packages/multimedia/video-codec-sdk) [`ffmpeg`](/packages/multimedia/ffmpeg) [`opencv`](/packages/cv/opencv) [`pybind11`](/packages/build/pybind11) [`ros:jazzy-desktop`](/packages/robots/ros) [`vpi`](/packages/cv/vpi) [`isaac-ros:common-3.2-jazzy-desktop`](/packages/robots/isaac-ros) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras`](/home/narandill/Projects/fph/Internal-AI-Base-Station-Server/third_party/jetson-containers/packages/robots/ros/Dockerfile.ros2.extras) |

</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag isaac-ros)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host isaac-ros:36.4.0

```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag isaac-ros)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag isaac-ros) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build isaac-ros
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
