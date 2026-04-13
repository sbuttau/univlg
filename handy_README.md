# Where things are and how to find them
# Container
```bash
export HOST_REPO="/mnt/projects/dm/warm_up/u0205/univlg"
singularity exec --nv --bind "$HOST_REPO:/workspaces" "univlg.sif" bash
source ~/.venvs/univlg/bin/activate
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH
```

## Datasets and dataloaders
dataset classes are in `univlg/data_video/dataset_mapper_language.py`

- frames_square_highres must be inside the 'data/' folder

## EVALUATION
You need to store some data paths to perform an inference run. To do so, run an eval with only a bunch of data and store it.
Data is stored by default under `ckpts/misc`.

(some variables need to be hardcoded in main.sh)

| NOTE: to use subsample data you need to set `DATASETS.TEST.SUMBSAMPLE_DATA` and DATASETS.TEST_SUBSAMPLED in `univlg/config.py`: `add_mask2former2_video_config`.

Before starting the scripts, make sure you logged in on wandb: 
```bash 
wandb login
```
Ex.
```python
cfg.TEST.SUBSAMPLE_DATA = 2
cfg.TRAIN_SUBSAMPLE_DATA = None
cfg.DATASETS.TEST_SUBSAMPLED = ['scanrefer_scannet_anchor_val_single_batched']
```
First run eval to store data sample.

NOTE: model alone takes 2GB, running the code on 2 scenes takes ~25GB RAM, ~5GB GPU VRAM.

```bash
export CKPT_PATH="ckpts/univlg.pth"
export SCANNET_DATA_DIR="/workspaces/univlg/data/mask3d_processed/scannet/two_scene_database.yaml" # this is not used
export SCANNET_200_DATA_DIR="/workspaces/univlg/data/mask3d_processed/scannet200/train_database.yaml"
source scripts/setup.sh
configure_local
NUM_VAL_DATALOADERS=1 NUM_DATALOADERS=1 EVAL_ONLY=1 RETURN_SCENE_BATCH_SIZE=1 \
TEST_DATASET_INFERENCE=True \
TEST_RESULT_EXPORT_PATH="$OUTPUT_DIR/test_results" \
SCANNET_DATA_DIR="$SCANNET_DATA_DIR" \
SCANNET200_DATA_DIR="$SCANNET200_DATA_DIR" \
VISUALIZE_REF=True \
VIZ_EXTRA_REF=True \
VISUALIZE_LOG_DIR="outputs/viz_ref" \
$PREFIX "${PREFIX_ARGS[@]}" scripts/main.sh \
DATASETS.TEST "('scanrefer_scannet_val_sentence_test_one_batched',)" \
SAVE_DATA_SAMPLE False \
DATA_SAMPLE_PATH ckpts/misc/long_sentence_test/one_sentence
```
TODO: need to check if RAM loading can be lightened

Once the data is stored, you can run a standalone eval to visualize your results:
```bash
export CKPT_PATH="ckpts/univlg.pth"
source scripts/setup.sh
configure_local
USE_STANDALONE=1 $PREFIX "${PREFIX_ARGS[@]}" scripts/main.sh \
USE_AUTO_NOUN_DETECTION False \
USE_SEGMENTS False
```
it will save an output folder with the instructions to open the visualizer online (you should do `cd to your folder` and then `python -m http.server 6008` or to your favorite port).

Options for datasets are:
- `sr3d_ref_scannet_val_single_batched`
- `scanrefer_scannet_anchor_val_single_batched`
- `nr3d_ref_scannet_anchor_val_single_batched`
- `matterport_train_single`
- `scannet200_context_instance_train_200cls_single_highres_100k`


## Model components
- vision backbone (DINOv2): `UniVLGVisualBackbone` (`univlg/modeling/visual_backbone.py`)
- decoder: `VideoMultiScaleMaskedTransformerDecoder` (`univlg/modeling/transformer_decoder/video_mask2former_transformer_decoder.py`)
- JINA text tokenizer: `univlg/data_video/dataset_mapper_language.py: 157
- JINA text encoder:  `univlg/data_video/dataset_mapper_language.py: 65