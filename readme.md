# A Resource for Evaluating Creative Search on the Infinite Index

## Accessing the dataset

Currently, the dataset is hosted here: [https://files.webis.de/data-in-progress/data-research/multimodality/topics-for-generative-models-review/index.html](https://files.webis.de/data-in-progress/data-research/multimodality/topics-for-generative-models-review/index.html) (will be moved to Zenodo)

The file structure of the data.zip can be integrated with this repository.

To simply access the textual image descriptions (to form *iterative topics*), see the `[reddit|pexels|lexica]/doccano_*_single_image.jsonl` files.


## Recreating the dataset

To reproduce the full extraction pipeline, the `reddit/comments.jsonl` and `reddit/submissions.jsonl` need to be extracted from previously downloaded [Pushshift dataset](https://ojs.aaai.org/index.php/ICWSM/article/view/7347/7201) using the script described in `pushshift/`.

The following steps should then be applied to create the provided dataset of r/captionthis data:
- Match top-level threads with image urls and comments and download the images using the `reddit/match_comments_submissions.py` script.
- Remove meta-comments using the `reddit/remove_by_blocklist.py` script.
- De-duplicate the images (robustly) using the `reddit/remove_duplicates.py` script.
- One Imgur soft-404 image (containing the text "This image is not available" as seen in `reddit/output/removed_404`) was moved manually.
- Build an index of text and image embeddings for the nearest neighbor search using the `reddit/build_index.py` script.
- Prepare the Doccano datasets (for the later annotation) using the `reddit/generate_doccano_*.py` scripts.

For the Pexels dataset:
- Download the [Pexels dataset](https://github.com/cj-mills/pexels-dataset) (768p source images) and unzip the images into a directory. The path needs to be specified at the top of the following Python scripts.
- Build an index of text and image embeddings for the nearest neighbor search using the `pexels/build_index.py` script.
- Prepare the Doccano datasets (for the later annotation) using the `pexels/generate_doccano_*.py` scripts.

For the Lexica dataset:
- The dataset will automatically be downloaded via [Huggingface](https://huggingface.co/datasets/vera365/lexica_dataset).
- Build an index of text and image embeddings for the nearest neighbor search using the `lexica/build_index.py` script.
- Prepare the Doccano datasets (for the later annotation) using the `lexica/generate_doccano_*.py` scripts.

## Hosting the dataset images

The `image_stitch_server.py` script can be used to host a web server that serves the images downloaded from each of the datasets. The url is given as `hostname:port/dataset/imgname.jpg(+dataset/imgname.jpg)*` so that one or multiple images can be displayed from a single url. This will be helpful for the Doccano annotation (as described below). The hostname under which the images are available must be adjusted in the other Python scripts so that the urls are correctly represented in the Doccano datasets.

## Annotation experiments

The `[reddit|pexels|lexica]/*.jsonl` files can be imported into [Doccano](https://github.com/doccano/doccano) as DocumentClassification tasks. If the images are hosted via the url specified in the `.jsonl` files (as described above), they will be displayed in Doccano via the [`im_url` key](https://github.com/doccano/doccano/pull/1430).
By importing the labels specified in the `two_label_config.json`, they match the numbers and colors shown with pairs of images.

The annotation experiment described in the paper is based on the `doccano_*_closest_clip_match_by_comment.jsonl` files, which show the nearest-neighbor construction described in the paper (in contrast to the `..._image.jsonl` files, which choose the nearest neighbors based on image similarity).

Using the `annotation/create_gallery.py` script, a html page displaying the images by the classes implied by the annotation results can be created.
