# robopoint

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`robopoint`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36.2']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`torch`](/packages/ml/torch) [`torchvision`](/packages/ml/torchvision) [`transformers`](/packages/llm/transformers) [`bitsandbytes`](/packages/llm/bitsandbytes) [`numpy`](/packages/numeric/numpy) [`gradio`](/packages/ui/gradio) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`robopoint`]() `(2025-03-06, 12GB)` |
| &nbsp;&nbsp;&nbsp;Models | [`robopoint-v1-vicuna-v1.5-13b`](https://huggingface.co/wentao-yuan/robopoint-v1-vicuna-v1.5-13b) `(2024-09-22, 25GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`robopoint`]() | `2025-03-06` | `arm64` | `12GB` |
<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
#download the model to the Jetson SSD from huggingface (preferred)
git clone https://huggingface.co/wentao-yuan/robopoint-v1-vicuna-v1.5-13b /data/models

# automatically pull or build a compatible container image and mount the model path to the container
jetson-containers run --volume /data/models:/data/models $(autotag robopoint)

#build the image 
jetson-containers build robopoint