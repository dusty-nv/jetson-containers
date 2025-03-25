# genesis-world

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)

docs.md
<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`genesis-world`** |                                                                                                                                                                                                                                                                                                                                                      |
| :-- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=36.1.0']`                                                                                                                                                                                                                                                                                                                                   |
| &nbsp;&nbsp;&nbsp;Dependencies | [`rust`](/packages/ml/rust) [`cmake`](/packages/build/cmake) [`torch:2.6`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`torchaudio`](/packages/pytorch/torchaudio) [`vtk`](/packages/vtk) [`pymeshlab`](/packages/pymeshlab) [`taichi`](/packages/taichi) [`splashsurf`](/packages/splashsurf) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile)                                                                                                                                                                                                                                                                                                                           |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/genesis-world:r36.4.0`](https://hub.docker.com/r/dustynv/genesis-world/tags) `(2025-03-18, 23.6GB)`                                                                                                                                                                                                                                       |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag |     Date     | Arch |   Size    |
| :-- |:------------:| :--: |:---------:|
| &nbsp;&nbsp;[`dustynv/genesis-world:r36.4.0`](https://hub.docker.com/r/dustynv/genesis-world/tags) | `2025-03-18` | `arm64` | `23.6GB` |


</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag genesis-world)

# or explicitly specify one of the container images above
jetson-containers run dustynv/genesis-world:r36.4.0

# or if using 'docker run' (specify image and mounts/etc)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/genesis-world:r36.4.0
