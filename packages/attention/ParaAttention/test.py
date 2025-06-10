#!/usr/bin/env python3

print("Testing ParaAttention...")
import torch
import torch.distributed as dist
from diffusers import WanPipeline

# from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
# model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# flow shift should be 3.0 for 480p images, 5.0 for 720p images
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
pipe.to("cuda")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

parallelize_pipe(
    pipe,
    mesh=init_context_parallel_mesh(
        pipe.device.type,
    ),
)

# from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
#
# apply_cache_on_pipe(pipe)

# Enable memory savings
# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
# pipe.enable_vae_tiling()

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

video = pipe(
    prompt="An astronaut dancing vigorously on the moon with earth flying past in the background, hyperrealistic",
    negative_prompt="",
    height=480,
    width=832,
    num_frames=1,
    num_inference_steps=30,
    output_type="pil" if dist.get_rank() == 0 else "pt",
).frames[0]

if dist.get_rank() == 0:
    print("Saving video to wan.mp4")
    export_to_video(video, "wan.mp4", fps=15)

dist.destroy_process_group()
print('ParaAttention OK\n')