# xgrammar

> [`BUILD`](#user-content-build)

<details open>
<summary><b><a id="build">BUILD CONTAINER</b></summary>
<br>

If you use [`autotag`](/docs/run.md#autotag) as shown above, it'll ask to build the container for you if needed.  To manually build it, first do the [system setup](/docs/setup.md), then run:
```bash
jetson-containers build xgrammar
```
The dependencies from above will be built into the container, and it'll be tested during.  Run it with [`--help`](/jetson_containers/build.py) for build options.
</details>

To use xGrammar with vLLM 0.6.6 comment the following lines in `vllm/model_executor/guided_decoding/__init__.py`:

```py
if current_platform.get_cpu_architecture() is not CpuArchEnum.X86:
    logger.warning("xgrammar is only supported on x86 CPUs. "
                    "Falling back to use outlines instead.")
    guided_params.backend = "outlines"
```
