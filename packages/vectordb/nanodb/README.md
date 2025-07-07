# nanodb

> [`CONTAINERS`](#user-content-containers) [`IMAGES`](#user-content-images) [`RUN`](#user-content-run) [`BUILD`](#user-content-build)


<a href="https://youtu.be/ayqKpQNd1Jw"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/nanodb_horse.gif"></a>

NanoDB is a CUDA-optimized multimodal vector database that uses embeddings from the [CLIP](https://openai.com/research/clip) vision transformer for txt2img and img2img similarity search. The [demo video](https://youtu.be/ayqKpQNd1Jw) above is running in realtime on 275K images from the MS COCO image captioning dataset using Jetson AGX Orin, and shows a fundamental capability in multimodal applications - operating in a shared embedding space between text/images/etc., and being able to query with a deep contextual understanding. 

In addition to effectively indexing and searching your data at the edge, these vector databases are often used in tandem with LLMs for [Retrieval Augmented Generation](https://www.promptingguide.ai/techniques/rag) (RAG) for long-term memory beyond their built-in context length (4096 tokens for Llama-2 models), and Vision-Language Models also use the same embeddings as inputs. 

### Indexing Data

NanoDB can recursively scan directories of images, compute their CLIP embeddings, and save them to disk in float16 format.  To ingest content into the database, start the container with the path to your dataset mounted (or store your dataset under `jetson-containers/data`, which is automatically mounted into the container under `/data`)  And run the `nanodb --scan` command:

```bash
./run.sh -v /path/to/your/dataset:/my_dataset $(./autotag nanodb) \
  python3 -m nanodb \
    --scan /my_dataset \
    --path /my_dataset/nanodb \
    --autosave --validate 
```
> To download a pre-indexed database and skip this step, see the [**NanoDB tutorial**](https://www.jetson-ai-lab.com/tutorial_nanodb.html) on Jetson AI Lab

* `--scan` optionally specifies a directory to recursively scan for images
  * Supported image extensions are:  `'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'`
  * You can specify `--scan` multiple times to import different directories
* `--path` specifies the directory that the NanoDB config/database will be saved to or loaded from
  * This directory will be created for a new database if it doesn't already exist.
  * If there's already an existing NanoDB there, it will be loaded first, and any scans performed are added to that database.
  * After images have been added, you can launch NanoDB with `--path` only to load a ready database.
* `--autosave` automatically saves the NanoDB embedding vectors after each scan, and after every 1000 images in the scan.
* `--validate` will cross-check each image against the database to confirm that it returns itself (or finds duplicates already included)

Only the embedding vectors are actually saved in the NanoDB database - the images themselves should be retained elsewhere if you still wish to view them.  The original images are not needed for search/retrieval after the indexing process - they're only needed for human viewing.

### Console Commands

Once the database has loaded and completed any start-up operations like `--scan` or `--validate`, it will drop down to a `>` prompt from which the user can run search queries, perform additional scans, and save the database from the terminal:

```bash
> a girl riding a horse

* index=80110   /data/datasets/coco/2017/train2017/000000393735.jpg      similarity=0.29991915822029114
* index=158747  /data/datasets/coco/2017/unlabeled2017/000000189708.jpg  similarity=0.29254037141799927
* index=123846  /data/datasets/coco/2017/unlabeled2017/000000026239.jpg  similarity=0.292171448469162
* index=127338  /data/datasets/coco/2017/unlabeled2017/000000042508.jpg  similarity=0.29118549823760986
* index=77416   /data/datasets/coco/2017/train2017/000000380634.jpg      similarity=0.28964102268218994
* index=51992   /data/datasets/coco/2017/train2017/000000256290.jpg      similarity=0.28929752111434937
* index=228640  /data/datasets/coco/2017/unlabeled2017/000000520381.jpg  similarity=0.28642547130584717
* index=104819  /data/datasets/coco/2017/train2017/000000515895.jpg      similarity=0.285491943359375
```

> By default, searches return the top 8 results, but you can change this with the `--k` command-line argument when starting nanodb.

To scan additional items after the database has started, enter the path to your dataset directory or file:

```bash
> /data/pascal_voc

-- loaded /data/pascal_voc/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg in 4 ms
-- loaded /data/pascal_voc/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg in 2 ms
-- loaded /data/pascal_voc/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg in 3 ms
-- loaded /data/pascal_voc/VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg in 3 ms

...
```

To save the updated database with any changes/additions made, use the `save` command:

```
> save
-- saving database to /my_dataset/nanodb
```

### Interactive Web UI

To spin up the Gradio server, start nanodb with the `--server` command-line argument:

```bash
./run.sh -v /path/to/your/dataset:/my_dataset $(./autotag nanodb) \
  python3 -m nanodb \
    --path /my_dataset/nanodb \
    --server --port=7860
```
> The default port is 7860, bound to all network interfaces `(--host=0.0.0.0)`

Then navigate your browser to `http://HOSTNAME:7860?__theme=dark`, and you can enter text search queries as well as drag/upload images:

<a href="https://youtu.be/ayqKpQNd1Jw"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/nanodb_tennis.jpg"></a>

<details open>
<summary><b><a id="containers">CONTAINERS</a></b></summary>
<br>

| **`nanodb`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T ['>=32.6']` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build/build-essential) [`pip_cache:cu126`](/packages/cuda/cuda) [`cuda:12.6`](/packages/cuda/cuda) [`cudnn:9.3`](/packages/cuda/cudnn) [`python`](/packages/build/python) [`numpy`](/packages/numeric/numpy) [`cmake`](/packages/build/cmake/cmake_pip) [`onnx`](/packages/ml/onnx) [`pytorch:2.8`](/packages/pytorch) [`cuda-python`](/packages/cuda/cuda-python) [`faiss`](/packages/vectordb/faiss) [`faiss_lite`](/packages/vectordb/faiss_lite) [`torchvision`](/packages/pytorch/torchvision) [`huggingface_hub`](/packages/llm/huggingface_hub) [`rust`](/packages/build/rust) [`transformers`](/packages/llm/transformers) [`tensorrt`](/packages/cuda/tensorrt) [`torch2trt`](/packages/pytorch/torch2trt) [`clip_trt`](/packages/vit/clip_trt) |
| &nbsp;&nbsp;&nbsp;Dependants | [`local_llm`](/packages/llm/local_llm) [`nano_llm:24.4`](/packages/llm/nano_llm) [`nano_llm:24.4-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4-humble`](/packages/llm/nano_llm) [`nano_llm:24.4-iron`](/packages/llm/nano_llm) [`nano_llm:24.4.1`](/packages/llm/nano_llm) [`nano_llm:24.4.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.4.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.4.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.4.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.5`](/packages/llm/nano_llm) [`nano_llm:24.5-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5-humble`](/packages/llm/nano_llm) [`nano_llm:24.5-iron`](/packages/llm/nano_llm) [`nano_llm:24.5.1`](/packages/llm/nano_llm) [`nano_llm:24.5.1-foxy`](/packages/llm/nano_llm) [`nano_llm:24.5.1-galactic`](/packages/llm/nano_llm) [`nano_llm:24.5.1-humble`](/packages/llm/nano_llm) [`nano_llm:24.5.1-iron`](/packages/llm/nano_llm) [`nano_llm:24.6`](/packages/llm/nano_llm) [`nano_llm:24.6-foxy`](/packages/llm/nano_llm) [`nano_llm:24.6-galactic`](/packages/llm/nano_llm) [`nano_llm:24.6-humble`](/packages/llm/nano_llm) [`nano_llm:24.6-iron`](/packages/llm/nano_llm) [`nano_llm:24.7`](/packages/llm/nano_llm) [`nano_llm:24.7-foxy`](/packages/llm/nano_llm) [`nano_llm:24.7-galactic`](/packages/llm/nano_llm) [`nano_llm:24.7-humble`](/packages/llm/nano_llm) [`nano_llm:24.7-iron`](/packages/llm/nano_llm) [`nano_llm:main`](/packages/llm/nano_llm) [`nano_llm:main-foxy`](/packages/llm/nano_llm) [`nano_llm:main-galactic`](/packages/llm/nano_llm) [`nano_llm:main-humble`](/packages/llm/nano_llm) [`nano_llm:main-iron`](/packages/llm/nano_llm) |
| &nbsp;&nbsp;&nbsp;Dockerfile | [`Dockerfile`](Dockerfile) |
| &nbsp;&nbsp;&nbsp;Images | [`dustynv/nanodb:r35.2.1`](https://hub.docker.com/r/dustynv/nanodb/tags) `(2023-12-14, 6.9GB)`<br>[`dustynv/nanodb:r35.3.1`](https://hub.docker.com/r/dustynv/nanodb/tags) `(2023-12-15, 7.0GB)`<br>[`dustynv/nanodb:r35.4.1`](https://hub.docker.com/r/dustynv/nanodb/tags) `(2023-12-12, 6.9GB)`<br>[`dustynv/nanodb:r36.2.0`](https://hub.docker.com/r/dustynv/nanodb/tags) `(2024-03-08, 7.8GB)`<br>[`dustynv/nanodb:r36.4.0`](https://hub.docker.com/r/dustynv/nanodb/tags) `(2025-01-14, 7.2GB)` |

</details>

<details open>
<summary><b><a id="images">CONTAINER IMAGES</a></b></summary>
<br>

| Repository/Tag | Date | Arch | Size |
| :-- | :--: | :--: | :--: |
| &nbsp;&nbsp;[`dustynv/nanodb:r35.2.1`](https://hub.docker.com/r/dustynv/nanodb/tags) | `2023-12-14` | `arm64` | `6.9GB` |
| &nbsp;&nbsp;[`dustynv/nanodb:r35.3.1`](https://hub.docker.com/r/dustynv/nanodb/tags) | `2023-12-15` | `arm64` | `7.0GB` |
| &nbsp;&nbsp;[`dustynv/nanodb:r35.4.1`](https://hub.docker.com/r/dustynv/nanodb/tags) | `2023-12-12` | `arm64` | `6.9GB` |
| &nbsp;&nbsp;[`dustynv/nanodb:r36.2.0`](https://hub.docker.com/r/dustynv/nanodb/tags) | `2024-03-08` | `arm64` | `7.8GB` |
| &nbsp;&nbsp;[`dustynv/nanodb:r36.4.0`](https://hub.docker.com/r/dustynv/nanodb/tags) | `2025-01-14` | `arm64` | `7.2GB` |

> <sub>Container images are compatible with other minor versions of JetPack/L4T:</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R32.7 containers can run on other versions of L4T R32.7 (JetPack 4.6+)</sub><br>
> <sub>&nbsp;&nbsp;&nbsp;&nbsp;• L4T R35.x containers can run on other versions of L4T R35.x (JetPack 5.1+)</sub><br>
</details>

<details open>
<summary><b><a id="run">RUN CONTAINER</a></b></summary>
<br>

To start the container, you can use [`jetson-containers run`](/docs/run.md) and [`autotag`](/docs/run.md#autotag), or manually put together a [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command:
```bash
# automatically pull or build a compatible container image
jetson-containers run $(autotag nanodb)

# or explicitly specify one of the container images above
jetson-containers run dustynv/nanodb:r36.4.0

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host dustynv/nanodb:r36.4.0
```
> <sup>[`jetson-containers run`](/docs/run.md) forwards arguments to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some defaults added (like `--runtime nvidia`, mounts a `/data` cache, and detects devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
jetson-containers run -v /path/on/host:/path/in/container $(autotag nanodb)
```
To launch the container running a command, as opposed to an interactive shell:
```bash
jetson-containers run $(autotag nanodb) my_app --abc xyz
```
You can pass any options to it that you would to [`docker run`](https://docs.docker.com/engine/reference/commandline/run/), and it'll print out the full command that it constructs before executing it.
</details>
<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build nanodb
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>
