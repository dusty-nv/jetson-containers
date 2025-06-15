## Original Work

This package was **heavily** insprired by the `vlm` service of jetson-platform-services repo.
https://github.com/NVIDIA-AI-IOT/jetson-platform-services/blob/ad7d3017f7ec75540b494fe1b61552a59b0b73a3/inference/vlm/README.md

The main changes are
- It conforms **OpenAI Vision API**, so it can work with a tool like **n8n**.
- It can take a **local video device** (e.g., USB webcam) as an image source, by providing `v4l2://` path in the `url` field inside `image_url`.

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
  <summary>(Click to expand) A big table indicating model readiness</summary>
<pre><code>
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
</code></pre>
</details>

### API Endpoint

| Method | API Endpoint | Description |
| ------ | ------------ | ----------- |
| `GET`  | `/v1/models` | return the VLM name in list |
| `POST` | `/v1/chat/completion` | Chat with the VLM using OpenAI style chat completions. Supports referencing added streams in the prompts. |
| `POST` | `/v1/alerts`          | Set an alert prompt the VLM will evaluate continuously on the input live stream. Can be used to trigger notifications when alert states are true. |

You can check the documentation of (the original) VLM Service of Jetson Platform Services as a reference.
https://docs.nvidia.com/jetson/jps/inference-services/vlm.html#overview

### OpenAI Vision API compatible endpoint

#### Using a local USB webcam as video stream input

> [OpenAI Vision API - Passing a URL](https://platform.openai.com/docs/guides/images?api-mode=chat&format=url)

You can specify the local v4l2 device path after `v4l2://` and put it in the `image_url`'s `url` field.

The v4l2 device is registered as the input video stream and VLM processes the latest frame of the video stream.

```bash
curl --location --request POST 'http://0.0.0.0:9000/v1/chat/completions' \
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
          "type": "text",
          "text": "Describe the scene."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "v4l2:///dev/video0"
          }
        }
      ]
    }
  ],
  "max_tokens": 128
}'
```

#### RTSP output for preview

Once you use a v4l2 video source, it is regsitered as the input stream.

You can view the output RTSP stream on `rtsp://<IP_ADDR>:5011/out`.

![](https://github.com/user-attachments/assets/b8e270b5-6b71-4988-b8f6-fd0e549111eb)

#### How to adjust the focus

You can run `v4l2-ctl` on host while the camera is working inside the container.

```bash
sudo apt install v4l-utils
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-ctrls
v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=48
```

#### Passing a Base64 encoded image

> [OpenAI Vision API - Passing a Base64 encoded image](https://platform.openai.com/docs/guides/images?api-mode=chat&format=base64-encoded)

You can also embed the image and pass that through API.

<details>
  <summary>(Click to expand) curl command with base64 encoded image</summary>
<pre><code>
curl --location --request POST 'http://0.0.0.0:9000/v1/chat/completions' \
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
          "type": "text",
          "text": "Describe the scene."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEBLAEsAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAA1AFADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDjrQ+XbyuEyU5wO+K9F8KaFq2t+CNVt47N1e4dDGG46EVxWkyrbXIkZVcRsCVPQ1sat8RdVW2kgsppLSNCoCwnG4d65MK6ablPc78Y6jSjD1+42JPhlrtjZ+bMkAaRSdgkGRXNzaPdJcpHcwkPG+HXI5BGCaox6tPqQa4S/vTMepdycH3zVqxnmMga8mSRzwAD85qZRo3vFNMIVa1rSaZLd6DJZLkqFIHQnrW1PpB/4VzqTPt+0BwVjBySOKvW2kyalZ7VkQxbSjmR8BVPQkn0/P8ATO/F4Uhs444LjVo21CUZigjQsqgDueo+uKUH7OXPbo+pVSbqJR21R5HpNz9i0YpcxMoJPLLg1LZajaBciQAnsa9G+xWk802natAkqLnO4DKEd815sdMs4NVmjRN0QY7M+lc9RxZ1Urts0LPVrW2eRjKvzrisfxBqMV1DAkTqwV8mtux0u0eG5JiGQBjiuT1qzS3MRjXljiu2i5RjZeZzVuVyvLyNLzfKjlbGCD1rAeZru5ypwu75cegrTv3P2SbnrgCuYs52SRd/occ4rKirpsVd+8kb7XRmtlEhKo0m0hG2nH1rp/DVtFKJLmTaLWE42hwHfAz+XvXn9q73CxhVJO/cFXk9a6G2RcFyh2yoVOxym5eQeeh79QatpJ6mSu1odDq3jOO5tIrGzMcUPnLsigVsnnqWx8w6+oqrZeIL3T/E1vepNIQJiFVmzuAXHOe5GR+NM0a30zTLK5ZJJJJjMyebMw46AlQBx6Z/xqv4vgWzsdN1K2G54Zm3LjIwwGN2DnHGM+pFZT973TWD5FzM9G1fUpb/AE+2vpkeFrlTuwAMkHGf5enrXMiOBpSQS2fU1c8L3cepaXFb6ix+xMUn3O+DGpGTg+vOPQ1oalHpouVXToi0WcAsck/nWEafM7t7G/tVHRLcyrYNFb3DMCFbpWRrtmEjspCOW5rutWsYU0RRF949q5/xNAPK06NfvBQK7qFS90znrxTSaPOdYuxFaAn7rPtPt71hBhhHYjIbBr0/w14bZpJxrVvLbj+AMmc/zrc/4Q/TJRgS2+D2KgH+VTC8Va34iqNSle55z4V0uaeJ5iMRyZjQZwSvc/n/ACqz4nB0DTIYnZSqsdhJ555x+deo2HhSSOJY7S7VFA2gIE/wrn9e+FK6tdeff3l3O3QbpBhfoAMUWbd2hc6SsmeODWi80YeQbRIHJzxtwDj8SOfwrdsNeE1uBJOUlaNYFX0VTnntzgZrso/gpZmQBLi5XPo3T9K0rb4LJaTJNb6pdJIvQkqf5rU1JwS1TFTjNs5S6s7zT447ueFzHIRwQdrEdwBwMfSvVdAOlS2VsHdRciNS+R3xViPwrqCQCO4vYZ1Ax88OTj8CKtp4dkCqGdeAANkZHH4k1zwrqD2OiVFzW49008xbfNBFUbmDTppEZ5FJTpV19BYD7zkey1T/ALEwMky/pWyxi/lM3hn3NFJyx+ZQfrUnkwyD54kP4UUV277nLa2wye1hjizGm1vUVGsJEQIlfP1oorCpFXNYSZc07zFfPmMfrWrJI7x7WORRRXnVpNPc7qSVieKdtoBAOBTjcMR0FFFTGTLcUV7iVjGe1ZxOVwc/nRRW0dTJn//Z"
          }
        }
      ]
    }
  ],
  "max_tokens": 128
}'
</code></pre>
</details>

<details>
<summary>(Click to expand) More prompt examples</summary>
<h4>OCR</h4>
<pre><code>
curl --location --request POST 'http://0.0.0.0:9000/v1/chat/completions' \
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
          "type": "text",
          "text": "Read the texts printed on the box."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "v4l2:///dev/video0"
          }
        }
      ]
    }
  ],
  "max_tokens": 128
}'
</code></pre>
<img src="https://github.com/user-attachments/assets/20d2358b-cf8c-4228-850f-766ec8adefa5" alt="Description" width="640">

<h4>Counting</h4>
<pre><code>
curl --location --request POST 'http://0.0.0.0:9000/v1/chat/completions' \
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
          "type": "text",
          "text": "What kind of chairs do you see in the scene?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "v4l2:///dev/video0"
          }
        }
      ]
    }
  ],
  "max_tokens": 128
}'
</code></pre>
<img src="https://github.com/user-attachments/assets/5d9089ef-76d4-4c54-8472-e433689150ba" alt="Description" width="640">


<h4>Base64-encoded PNG</h4>
<pre><code>
curl --location --request POST 'http://0.0.0.0:9000/v1/chat/completions' \
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
          "type": "text",
          "text": "Describe this icon."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAYNJREFUOE9j3CEjU/efkbGakYGBjYEE8J+B4Rfj//+tjNtlZX+SqhlmD9iQHbKy/2EC3CoqDNxKSnB3fL5xg4GZi4uBS04OLvbxwgWGn69ewfkoBvBqazMYzJjBwCUvD1ZwKjQUrFmntxfM/3DuHMPF7GyGH0+fYjcAJGq+YQODgJERw59Pnxj2GRgwsAkKMtifOcPAyMjI8Gz9eobL+fkoIYXiAmQDXm7bxnAhIwOs2HLrVgY+XV3SDLhSWsrwdOVKsAEqJSUMynl5pBnw9vhxsDdAgF1cnEHAwIB4A5jY2BiOe3khAoqZmcHx3DmG1wcOEBcG744eZbjd3Y0SWHpTpjD8//uXOANutrQwfDhzBsUAqeBgBmE7O8IGGC1cyHA+KQlsGzJgExFhUK+pYbhcUIA7GkWdnBikwsIYHi1YwPD+xAkUhYLm5gzyiYkMT1evZni9dy/2hCRsY8PAwsfH8PPlS4YPZ8+iGMBvZMTAISHB8OfzZ4a3hw8jDKA8M1GYnQE8m7INTv0HFQAAAABJRU5ErkJggg=="
          }
        }
      ]
    }
  ],
  "max_tokens": 128
}'
</code></pre>

<h4>Online image</h4>
<pre><code>
curl --location --request POST 'http://0.0.0.0:9000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
  "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What''s in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            }
        ]
    }],
  "max_tokens": 128
}'
</code></pre>

<h4>CSI camera</h4>
<pre><code>
curl --location --request POST 'http://0.0.0.0:9000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
  "messages": [
    {
      "role": "system",
      "content": "Describe the scene."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe the scene."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "csi://0"
          }
        }
      ]
    }
  ],
  "max_tokens": 128
}'
</code></pre>

</details>

#### n8n example.

<a href="https://github.com/user-attachments/assets/de8cf4dd-ff3b-4607-9c38-6837f830b7c6"><img src="https://github.com/user-attachments/assets/de8cf4dd-ff3b-4607-9c38-6837f830b7c6" width="800"></a>

### Build Container

```bash
jetson-containers build vila-microservice
```