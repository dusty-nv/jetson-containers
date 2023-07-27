# l4t-text-generation

<details open>
<summary><b>CONTAINERS</b></summary>



| **`l4t-text-generation`** | |
| :-- | :-- |
| &nbsp;&nbsp;&nbsp;Requires | `L4T >=34.1.0` |
| &nbsp;&nbsp;&nbsp;Dependencies | [`build-essential`](/packages/build-essential) [`python`](/packages/python) [`numpy`](/packages/numpy) [`cmake`](/packages/cmake/cmake_pip) [`onnx`](/packages/onnx) [`pytorch`](/packages/pytorch) [`torchvision`](/packages/pytorch/torchvision) [`bitsandbytes`](/packages/llm/bitsandbytes) [`transformers`](/packages/llm/transformers) [`awq`](/packages/llm/awq) [`onnxruntime`](/packages/onnxruntime) [`optimum`](/packages/llm/optimum) [`auto-gptq`](/packages/llm/auto-gptq) [`gptq-for-llama`](/packages/llm/gptq-for-llama) [`text-generation-webui`](/packages/llm/text-generation-webui) [`rust`](/packages/rust) [`text-generation-inference`](/packages/llm/text-generation-inference) |

</details>

<details open>
<summary><b>RUN CONTAINER</b></summary>
</br>

[`run.sh`](/run.sh) adds some default `docker run` args (like `--runtime nvidia`, mounts a [`/data`](/data) cache, and detects devices)
```bash
# automatically pull or build a compatible container image
./run.sh $(./autotag l4t-text-generation)

# or if using 'docker run' (specify image and mounts/ect)
sudo docker run --runtime nvidia -it --rm --network=host l4t-text-generation:35.2.1

```
To mount your own directories into the container, use the [`-v`](https://docs.docker.com/engine/reference/commandline/run/#volume) or [`--volume`](https://docs.docker.com/engine/reference/commandline/run/#volume) flags:
```bash
./run.sh -v /path/on/host:/path/in/container $(./autotag l4t-text-generation)
```
To start the container running a command, as opposed to the shell:
```bash
./run.sh $(./autotag l4t-text-generation) my_app --abc xyz
```
</details>
<details open>
<summary><b>BUILD CONTAINER</b></summary>
</br>

If you use [`autotag`](/autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do this System Setup, then run:
```bash
./build.sh l4t-text-generation
```
The dependencies from above will be built into the container, and it'll be tested.  See [`./build.sh --help`](/jetson_containers/build.py) for build options.
</details>
