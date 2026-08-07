# A Resource for Evaluating Creative Search on the Infinite Index

This repository documents the code for our paper `A Resource for Evaluating Creative Search on the Infinite Index` (currently under review). It introduces Cranfield-style topics that are designed to evaluate iterative generative systems, which are systems that allow for iterative refinement of generated content through user feedback. One example for such a system would be "prompt engineering on Stable Diffusion".

To build such topics, this repository provides (1) the software needed to re-create our dataset of topics, (2) a validation experiment workflow and (3) a demonstration of how to evaluate iterative generative systems.

### Paper Link

(Our paper is currently under review.)

### FAIR Research

We make our research data *findable* (through our paper, metadata and identifiers), *accessible* (by hosting the dataset and its description on [Zenodo](https://doi.org/10.5281/zenodo.20574977)), *interoperable* (by using file formats such as json and jsonl) and *reusable* (by providing scripts in this repository to use and reproduce our data and by using permissive licenses).

## Our Dataset

Our dataset consists of three parts: (1) A set of topics formed by extracting the abstract concepts submitted to [r/captionthis](https://www.reddit.com/r/captionthis/) on Reddit, (2) a set of topics based on stock image descriptions from [Pexels](https://github.com/cj-mills/pexels-dataset) and (3) a set of topics built with text-to-image prompt from [Lexica](https://huggingface.co/datasets/vera365/lexica_dataset). Our code treats the three sets separately.

### Accessing the Dataset

The dataset associated with this repository can be found on Zenodo: https://doi.org/10.5281/zenodo.20574977

After downloading the zip file, it should be unpacked and integrated into the folder structure of this repository.

To simply access the textual image descriptions (to form *interactive topics*), see the `[reddit|pexels|lexica]/doccano_*_single_image.jsonl` files.

### Datasheet

A full description of the dataset file structure and its Datasheet can be found in the resource description on Zenodo (https://doi.org/10.5281/zenodo.20574977).


### Reproducing the Dataset

The following steps describe the process to reproduce the full dataset creation pipeline. However, the final evaluation application (as described below) only requires the compiled `.jsonl` files that we provide via Zenodo.

For the Reddit dataset:
- To reproduce the full extraction pipeline, the `reddit/comments.jsonl` and `reddit/submissions.jsonl` need to be extracted from the previously downloaded [Pushshift dataset](https://ojs.aaai.org/index.php/ICWSM/article/view/7347/7201) using the script described in `pushshift/`. The following steps should then be applied to create the provided dataset of r/captionthis data.
- Match top-level threads with image urls and comments and download the images using the `reddit/match_comments_submissions.py` script.
- Remove meta-comments using the `reddit/remove_by_blocklist.py` script.
- De-duplicate the images (robustly) using the `reddit/remove_duplicates.py` script.
- One Imgur soft-404 image (containing the text "This image is not available" as seen in `reddit/output/removed_404`) was moved manually.
- Build an index of text and image embeddings for the nearest neighbor search using the `reddit/build_index.py` script.
- Prepare the Doccano datasets (for the later annotation) using the `reddit/generate_doccano_*.py` scripts.

For the Pexels dataset:
- Download the [Pexels dataset](https://github.com/cj-mills/pexels-dataset) (768p source images) and unzip it into the directory `pexels/extracted` such that it contains the `pexels-prompts-pairs.json` file and the `images` directory.
- Build an index of text and image embeddings for the nearest neighbor search using the `pexels/build_index.py` script.
- Prepare the Doccano datasets (for the later annotation) using the `pexels/generate_doccano_*.py` scripts.

For the Lexica dataset:
- The dataset will automatically be downloaded via [Huggingface](https://huggingface.co/datasets/vera365/lexica_dataset).
- Build an index of text and image embeddings for the nearest neighbor search using the `lexica/build_index.py` script.
- Prepare the Doccano datasets (for the later annotation) using the `lexica/generate_doccano_*.py` scripts.

### Hosting the Dataset Images

The `image_stitch_server.py` script can be used to host a (local) web server that serves the images downloaded from each of the datasets. The url is given as `hostname:port/dataset/imgname.jpg(+dataset/imgname.jpg)*` so that one or multiple images can be displayed from a single url. This will be helpful for the Doccano annotation (as described below). The hostname under which the images are available must be adjusted in the other Python scripts so that the urls are correctly represented in the Doccano datasets. Currently, it is simply set to `localhost`.

## Annotation Experiments for Validating the Topics

We provide a workflow for the validation of interactive topics based on our dataset presented above.
The core component of this workflow is a pair of strong candidate images for a given topic. Then, we survey assessor agreement on pairwise ratings under a certain configuration of personas.


For a given *interactive topic*, one strong candidate image is given by the originally matched image.
To extract another strong candidate, we only consider images from the same dataset. This allows for comparability with the originally matched image. There are enough samples in all three datasets to find a sufficient candidate for each topic.
Our approach is to find the image that is the nearest neighbor of the given description text w.r.t. their CLIP embeddings, which can be computed for images and texts. By design, this candidate matches the given description text well. However, CLIP embeddings do not encode all of the subtext in text and images. Thus, on r/captionthis, this method often shows a noticeable difference to the originally assigned image, which allows for different personas to produce different ratings of the candidates.
In case the originally assigned image is the nearest neighbor of the given description text, the second-nearest neighbor is used.


### Experimental Setup

With our validation workflow, we evaluate whether (1) giving a persona description makes the assessment of iterative generative systems more consistent (in the sense of *system stability*) and (2) using different personas affects the result at all (*topic ambiguity*), which later allows to visit different branches of the iterative generative systems.
For our exemplary experiment, we define two personas, whose descriptions are given as follows:

> Kevin, 15, is a school student who likes to spend his spare time online. He is signed up to Reddit and 4chan, where he follows the latest meme trends. His taste in memes can be described as edgy, and he considers his humor quite dark.

> Barbara, 62, has children and grandchildren. She is not signed up on image boards such as 4chan or Reddit, but enjoys a good laugh from images sent to her via WhatsApp and Facebook. She likes to forward light-hearted images to her nephews.

Assessors are assigned one of the two personas and asked which of the two image candidates, from the perspective of the assigned persona, matches the given description text best. For multiple assessors, there are three cases:
- (1) Assessors disagree despite having the same persona. This suggests that the topic is under-specified.
- (2) Assessors agree despite having different personas. This suggests that the topic is not sufficiently dependent on the persona, which would be required to evaluate the core feature of iterative generative systems.
- (3) Identical personas yield agreement, while different personas yield disagreement. This is the desired case for interactive topics: E.g., one persona recognizes a certain underlying subtext, while the other does not.


### Results

For an initial agreement study, three assessors each annotated the first 100 samples for each of the three datasets with one of the two personas given above (swapping the personas after 50 samples). The results are summarized in the following table:

|                        | r/captionthis | Pexels | Lexica |
|------------------------|---------------|--------|--------|
| (1): Kevins disagree   |   8            |      10  |     13   |
| (1): Barbaras disagree |    17           |    12    |     15   |  
| (2): Everyone agrees   |     44          |     43   |     41   |  
| (3): Equal personas agree |        31       |   35     |   31     |  

The results show a notable difference between both personas in the agreement of assessors with the same persona only on r/captionthis, with a higher agreement for the persona Kevin. This suggests that personas work well if they are tailored to the domain. This result lays the foundation for a more large-scale experiment with individual persona pairs for each dataset sample, for which automated methods might be useful.

### Technical Implementation

The `[reddit|pexels|lexica]/*.jsonl` files can be imported into [Doccano](https://github.com/doccano/doccano) as DocumentClassification tasks. If the images are hosted via the url specified in the `.jsonl` files (as described above), they will be displayed in Doccano via the [`im_url` key](https://github.com/doccano/doccano/pull/1430).
By importing the labels specified in the `two_label_config.json`, they match the numbers and colors shown with pairs of images.

The annotation experiment described above is based on the `doccano_*_closest_clip_match_by_comment.jsonl` files, which show the nearest-neighbor construction described in the paper (in contrast to the `..._image.jsonl` files, which choose the nearest neighbors based on image similarity).

Using the `annotation/create_gallery.py` script, a html page displaying the images by the classes implied by the annotation results can be created.


## Evaluating Iterative Generative Systems

The goal of this evaluation is to compare different iterative generative systems (such as *query refinement on image search*, *prompt engineering for image generation*, or specialized systems such as the one introduced by [Deckers et al. (2024)](https://doi.org/10.24963/ijcai.2024/845)) despite the different ways that users interact with them to arrive at a final result. The *interactive topics* contributed with this resource form the basis of such an evaluation.

### Experimental Setup

One experiment involves a number of systems, human assessors, and different topics.
Evaluating systems using our *interactive topics* works similar to how interactive IR systems are evaluated, e.g., in [TREC](https://www.nist.gov/publications/trec-6-interactive-track-report), since we face similar issues:
As described by [Kelly (2009)](https://doi.org/10.1561/1500000012), having all assessors evaluate all systems on all topics is not feasible: Rotation and counterbalancing are required to mitigate the effect of assessors seeing one system before another; topic splitting allows to avoid biases between topics; and asking for relative judgments instead of absolute ratings helps to reduce scale bias.

