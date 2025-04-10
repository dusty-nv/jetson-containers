
OpenDroneMap container ([website](https://www.opendronemap.org/), [GitHub](https://github.com/OpenDroneMap)) for Jetson.  This uses CUDA for accelerated SIFT feature extraction during image alignment & registration, and during SFM reconstruction.

Example to start the container building maps from a directory of images (like from [here](https://github.com/OpenDroneMap/ODM#quickstart) in the OpenDroneMap documentation)

```
jetson-containers run \
  -v /home/user/data:/datasets \
  $(autotag opendronemap) \
    python3 /code/run.py \
      --project-path /datasets project
```
