
* Segment Anything from https://github.com/facebookresearch/segment-anything

The `sam` container has a default run command to launch Jupyter Lab with notebook directory to be `/opt/`

Use your web browser to access `http://HOSTNAME:8888`

### How to run Jupyter notebooks

Once you are on Jupyter Lab site, navigate to `notebooks` directory.

#### Automatic Mask Generator Example notebook

Open `automatic_mask_generator_example.ipynb`.

Create a cell below the 4th cell, with only the following line and execute.

```
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Then, start executing the following cells (cells below **Set-up**)

#### Predictor Example notebook

Open `predictor_example.ipynb`.

Make sure you have `sam_vit_h_4b8939.pth` checkpoint file saved under `notebooks` directory.

Then, start executing the cells below **Set-up**.

### Benchmark script

You can run the following command to run a benchmark script.

```
python3 benchmark.py --save sam.csv
```

Or for full options:

```
python3 benchmark.py \
  --images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg  https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg \
  --runs=1 --warmup=0 \
  --save sam.csv
```

Outputs are:

- `sam_benchmark_output.jpg` :
- `sam.csv` (optional) : 
