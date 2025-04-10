
* NanoSAM from https://github.com/NVIDIA-AI-IOT/nanosam/

### Run the basic usage example and copy the result to host

```
./run.sh $(./autotag nanosam) /bin/bash -c " \
  cd /opt/nanosam && \
  python3 examples/basic_usage.py \  
    --image_encoder=data/resnet18_image_encoder.engine \
    --mask_decoder=data/mobile_sam_mask_decoder.engine && \
  mv data/basic_usage_out.jpg /data/ \
  "
```

### Benchmark

```
 ./run.sh $(./autotag nanosam) /bin/bash -c " \
   cd /opt/nanosam && \
   python3 benchmark.py --run 3 -s /data/nanosam.csv && \
   mv data/benchmark_last_image.jpg /data/ \
   "
 ```