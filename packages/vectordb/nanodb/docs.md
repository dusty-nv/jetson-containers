
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
