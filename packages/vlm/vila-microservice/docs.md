## Original Work

This package was **heavily** insprired by the `vlm` service of jetson-platform-services repo.
https://github.com/NVIDIA-AI-IOT/jetson-platform-services/blob/ad7d3017f7ec75540b494fe1b61552a59b0b73a3/inference/vlm/README.md

The main change is that it can accept video sources other than RTSP, including V4L so you can use a USB webcam as the input stream.

## Usage Examples

### Start Microservice Server

> If you plan to use a USB Webcam, first attach the device to your Jetson and run the following.

Start the docker container to run the visualization script.

```bash
jetson-containers run $(autotag vila-microservice)
```

#### First time running the container

When running for the first time, it will pulls the VILA model from Hugging Face and build the MLC model.

Once done, you will see an output like the following.

<details>
  <summary>Click to expand</summary>
```text
┌────────────────────────────┬─────────────────────────────────────────────────────────────────────────────┐
│ _name_or_path              │ ./llm                                                                       │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ architectures              │ ['LlamaForCausalLM']                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ drop_path_rate             │ 0.0                                                                         │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ hidden_size                │ 2560                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ image_aspect_ratio         │ resize                                                                      │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ interpolate_mode           │ linear                                                                      │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ mm_hidden_size             │ 1152                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ mm_projector_lr            │                                                                             │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ mm_use_im_patch_token      │ False                                                                       │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ mm_use_im_start_end        │ False                                                                       │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ mm_vision_select_feature   │ cls_patch                                                                   │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ mm_vision_select_layer     │ -2                                                                          │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ model_dtype                │ torch.bfloat16                                                              │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ model_type                 │ llama                                                                       │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ num_video_frames           │ 8                                                                           │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ resume_path                │ ./vlm                                                                       │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ s2                         │ False                                                                       │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ s2_max_split_size          │ 336                                                                         │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ s2_scales                  │ 336,672,1008                                                                │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ transformers_version       │ 4.36.2                                                                      │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ tune_language_model        │ True                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ tune_mm_projector          │ True                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ tune_vision_tower          │ True                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ vision_resolution          │ -1                                                                          │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ name                       │ VILA1.5-3b                                                                  │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ api                        │ mlc                                                                         │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ mm_vision_tower            │ /data/models/huggingface/models--Efficient-Large-Model--VILA1.5-3b/snapshot │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ mm_projector_path          │ /data/models/huggingface/models--Efficient-Large-Model--VILA1.5-3b/snapshot │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ mm_projector_type          │ mlp_downsample                                                              │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ attention_bias             │ False                                                                       │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ attention_dropout          │ 0.0                                                                         │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ bos_token_id               │ 1                                                                           │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ eos_token_id               │ 2                                                                           │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ hidden_act                 │ silu                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ initializer_range          │ 0.02                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ intermediate_size          │ 6912                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ max_position_embeddings    │ 4096                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ model_max_length           │ 4096                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ num_attention_heads        │ 20                                                                          │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ num_hidden_layers          │ 32                                                                          │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ num_key_value_heads        │ 20                                                                          │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ pad_token_id               │ 0                                                                           │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ pretraining_tp             │ 1                                                                           │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ rms_norm_eps               │ 1e-05                                                                       │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ rope_scaling               │                                                                             │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ rope_theta                 │ 10000.0                                                                     │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ tie_word_embeddings        │ False                                                                       │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ tokenizer_model_max_length │ 4096                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ tokenizer_padding_side     │ right                                                                       │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ torch_dtype                │ bfloat16                                                                    │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ use_cache                  │ True                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ vocab_size                 │ 32000                                                                       │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ quant                      │ q4f16_ft                                                                    │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ type                       │ llama                                                                       │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ max_length                 │ 4096                                                                        │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ prefill_chunk_size         │ -1                                                                          │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ load_time                  │ 169.52555129816756                                                          │
├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ params_size                │ 1300.8330078125                                                             │
└────────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘
```
</details>

### API Endpoint

| API Endpoint | Description |
| ------------ | ----------- |
| `/api/v1/live-stream`     | Manage live streams the AI service has access to. |
| `/api/v1/chat/completion` | Chat with the VLM using OpenAI style chat completions. Supports referencing added streams in the prompts. |
| `/api/v1/alerts`          | Set an alert prompt the VLM will evaluate continuously on the input live stream. Can be used to trigger notifications when alert states are true. |

You can see the documentation of (the original) VLM Service of Jetson Platform Services as a reference.
https://docs.nvidia.com/jetson/jps/inference-services/vlm.html#overview

### 1. Add a USB Webcam Stream

You can start by adding an RTSP stream for the VLM to use with the following curl command. This will use the POST method on the live-stream endpoint.

```bash
curl --location 'http://0.0.0.0:5010/api/v1/live-stream' \
  --header 'Content-Type: application/json' \
  --data '{
    "liveStreamUrl": "v4l2:///dev/video0"
  }'
```

This returns a unique `stream_id` for the stream you added.<br>
Currently the VLM will only support 1 stream but in the future this API will allow for multi-stream support.

#### RTSP output for preview

At this point, you can view the output stream

![](https://github.com/user-attachments/assets/b8e270b5-6b71-4988-b8f6-fd0e549111eb)

### 2. Ask a Question

You can also ask open ended questions to the VLM using the chat completions endpoint.

```bash
curl --location 'http://0.0.0.0:5010/api/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data '{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "stream",
          "stream": {
            "stream_id": "aaa"
          }
        },
        {
          "type": "text",
          "text": "Can you describe the scene?"
        }
      ]
    }
  ],
  "min_tokens": 1,
  "max_tokens": 128
}'
```

Currently, the server does not care the `stream_id`, so you can provide any random letter.

#### Output

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "3 black rolling chairs."
      }
    }
  ]
}
```

#### Preview Stream

![](https://github.com/user-attachments/assets/5d9089ef-76d4-4c54-8472-e433689150ba)

### Build Container

```bash
jetson-containers build vila-microservice
```