# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from datetime import datetime
from pathlib import Path
import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
print("Debugger attached!")
import detectron2.utils.comm as comm
import ipdb
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyviz3d.visualizer as viz
import torch
import torch.distributed
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import _log_api_usage
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel
from torch_scatter import scatter_mean

import wandb
from univlg import add_maskformer2_config, add_maskformer2_video_config
from univlg.data_video.data_utils import get_multiview_xyz
from univlg.data_video.sentence_utils import convert_grounding_to_od_logits_ref
from univlg.modeling.backproject.backproject import multiscsale_voxelize
from univlg.utils.decoupled_utils import breakpoint_on_error
from univlg.utils.misc import nanmax, nanmin
import matplotlib.pyplot as plt
import re
warnings.filterwarnings("ignore")

st = ipdb.set_trace


def create_ddp_model(
    model, *, fp16_compression=False, find_unused_parameters=False, **kwargs
):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa W605
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(
        model, **kwargs, find_unused_parameters=find_unused_parameters
    )
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import \
            default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def get_color(max_value: int, colormap='spring'):
    colormap = plt.get_cmap('spring')  # Pink is 0, Yellow is 1
    colors = [mcolors.to_rgb(colormap(i / max_value)) for i in range(max_value)]  # Generate colors
    return (np.array(colors) * 255).astype(int).tolist()


def get_color_preds(max_value: int, colormap='ignored'):
    """
    Returns a list of RGB colors where the first color is red and 
    the subsequent colors are pink.
    """
    # Define Red and Pink in RGB format (0-255)
    red = [255, 0, 0]
    pink = [255, 105, 180]  # Hot Pink
    
    if max_value <= 0:
        return []
    
    # First box is red, all others are pink
    colors = [red] + [pink] * (max_value - 1)
    
    return colors

def box_xyzxyz_to_cxcyczwhd(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    x_c = 0.5 * (x0 + x1)
    y_c = 0.5 * (y0 + y1)
    z_c = 0.5 * (z0 + z1)
    w = x1 - x0
    h = y1 - y0
    d = z1 - z0
    return torch.stack([x_c, y_c, z_c, w, h, d], dim=-1)

def visualize_pc_masks_and_bbox(
    pc, color, captions=None, pred_bboxs=None, pred_masks=None, 
    mask_pc=None, data_dir=None, current_token="", rank_idx=None
):
    """
    Visualize a point cloud and its predicted bounding box.
    
    Parameters:
      pc: N x 3 numpy array representing the point cloud.
      color: N x 3 numpy array (0-255) with colors corresponding to each point.
      captions: List of strings containing the captions for each point cloud.
      pred_bboxs: M x 6 numpy array representing a bounding box in [xmin, ymin, zmin, xmax, ymax, zmax] format.
      pred_masks: M x N bool numpy array representing a mask for M objects.
      data_dir: (optional) Base directory to save the visualization.
      sample_name: (optional) Name of the sample (used to structure the output directory).
      inputs: (optional) List of dictionaries containing metadata (e.g., 'dataset_name').
    """
    pred_colors = get_color_preds(len(pred_bboxs))
    point_size = 25

    v = viz.Visualizer()
    v.add_points("RGB", pc, colors=color, alpha=0.8, visible=True, point_size=point_size)
    v.add_labels(
        'Labels',
        [captions],
        [np.array([1.0, 0.0, 0.0])],
        [np.array([255.0, 0.0, 0.0])],
        visible=True
    )
    z_max = pc[:, 2].max() + 1.0
    v.add_labels(
        'Token_Info',
        [f"TOKEN: {current_token}"],
        [np.array([0, 0, z_max])],
        [np.array([255, 255, 255])], # Bianco
        visible=True
    )
    # Convert predicted bounding box to center-size format and add to visualization
    if pred_bboxs is not None:
        pred_bboxs = torch.from_numpy(pred_bboxs) if isinstance(pred_bboxs, np.ndarray) else pred_bboxs
        pred_bboxs = box_xyzxyz_to_cxcyczwhd(pred_bboxs).cpu().numpy()
        
        for i in range(pred_bboxs.shape[0]):
            v.add_bounding_box(
                f"Boxes (Pred) {i}",
                position=pred_bboxs[..., :3][i],
                size=pred_bboxs[..., 3:][i],
                color=np.array(pred_colors[i]),
                alpha=0.8,
                visible=(i == 0), 
                edge_width=0.03
            )
           
    if pred_masks is not None:
        for i in range(pred_masks.shape[0]):
            if mask_pc[pred_masks[i]].shape[0] == 0:
                print(f"Mask {i} is empty")
                continue

            v.add_points(
                f"Masks (Pred) {i}",
                mask_pc[pred_masks[i]],
                colors=np.array(pred_colors[i])[None].repeat(mask_pc[pred_masks[i]].shape[0], axis=0),
            )

    data_dir = Path(data_dir)
    datetime_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S.%f")[:-3]
    data_dir = data_dir / datetime_str
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saved to {data_dir}")
    v.save(str(data_dir))


def load_3d_data(cfg, batched_inputs, images_shape, device):
    valids = None
    multiview_data = None
    bs, v = images_shape[:2]
    
    multiview_data = {}
    multiview_data["multi_scale_xyz"] = [
        torch.stack(
            [batched_inputs[i]["multi_scale_xyz"][j] for i in range(bs)], dim=0
        ).to(device)
        for j in range(len(batched_inputs[0]["multi_scale_xyz"]))
    ]

    voxel_size = cfg.INPUT.VOXEL_SIZE[::-1]

    if cfg.INPUT.VOXELIZE:
        multiview_data["multi_scale_p2v"] = multiscsale_voxelize(
            multiview_data["multi_scale_xyz"], voxel_size
        )
    return valids, multiview_data

def process_lang_data(cfg, tokenizer, dataset_dict):
    from univlg.data_video.datasets.ref_coco_utils import (consolidate_spans,
                                                           get_root_and_nouns)
    from univlg.data_video.sentence_utils import (create_positive_map,
                                                  get_positive_tokens)
    
    utterance = dataset_dict["utterance"].lower()
    target_id = dataset_dict["target_id"]
    anchor_ids = dataset_dict['anchor_ids']
    anchor_names = dataset_dict['anchors_types']

    if dataset_dict.get("target_str", None) is not None:
        tokens_positive = get_positive_tokens(utterance, [dataset_dict["target_str"]])
    else:
        _, _, root_spans_spacy, _ = get_root_and_nouns(utterance)
        tokens_positive = [consolidate_spans(root_spans_spacy, utterance)]

    dataset_dict["tokens_positive"] = tokens_positive

    max_len = (
        cfg.MODEL.MAX_SEQ_LEN
        if not cfg.TEXT_ENCODER_TYPE == "clip"
        else 77
    )
    tokenized = tokenizer(
        utterance, return_tensors="pt", max_length=max_len, truncation=True
    )

    positive_map = create_positive_map(
        tokenized, tokens_positive, max_query_len=max_len
    )

    return {
        "text_caption": utterance,
        "target_id": target_id,
        "anchor_ids": anchor_ids,
        "anchors_names": anchor_names,
        "tokens_positive": tokens_positive,
        "tokenized": tokenized,
        "positive_map": positive_map,
        "positive_map_od": None,
        "annotation_id": dataset_dict['ann_id'] if "ann_id" in dataset_dict else None
    }

def get_pred_logits(cfg, lang_data, outputs):
    bs = len(lang_data)
    num_classes = max([len(lang_data[i]['anchor_ids']) + 1 for i in range(len(lang_data))]) # Root noun + anchor ids
    outputs["pred_logits"] = outputs["pred_logits"].sigmoid()
    reduce = "mean"
    outputs["pred_logits"] = torch.cat(
        [
            convert_grounding_to_od_logits_ref(
                logits=outputs["pred_logits"][i][None],
                num_class=num_classes + 1,
                positive_maps=lang_data[i]["positive_map"],
                reduce=reduce,
            )
            for i in range(bs)
        ]
    )
    outputs["pred_scores"] = outputs["pred_logits"]
    return outputs

# Function to split text into logical blocks (Keep it simple and robust)
def get_logical_chunks(text):
    # We split before keywords to keep them in the chunk they describe
    markers = ["with", "it does not", "it doesn't", "near", "on it", "and"]
    pattern = "|".join([f"(?={re.escape(m)})" for m in markers])
    chunks = [c.strip() for c in re.split(pattern, text, flags=re.IGNORECASE) if c.strip()]
    return chunks

# Function to map chunk words to model token indices
def get_chunk_indices(chunk_text, tokens_human):
    clean_chunk = chunk_text.lower().split()
    indices = []
    for i, t in enumerate(tokens_human):
        # Remove the RoBERTa/GPT 'Ġ' symbol and spaces
        clean_t = t.replace('Ġ', '').strip().lower()
        if clean_t in clean_chunk and clean_t != '':
            indices.append(i)
    return indices

def get_pred_boxes(outputs, scannet_pc, max_valid_points):
    masks = outputs['pred_masks'] > 0

    # remove padded tokens
    for j in range(len(max_valid_points)):
        masks[j, :, max_valid_points[j]:] = False

    pc = scannet_pc
    pc = pc[:, None].repeat(1, masks.shape[1], 1, 1)
    pc[torch.where(masks == 0)] = torch.nan

    boxes = torch.cat([
        nanmin(pc, dim=2)[0], nanmax(pc, dim=2)[0]
    ], 2)

    # if only one point is in the mask, the box will still be too small
    boxes[masks.sum(2) <= 1] = torch.tensor([0, 0, 0, 1e-2, 1e-2, 1e-2], device=boxes.device)
    outputs['pred_boxes'] = boxes
    return outputs


def get_dummy_data(cfg, device):
    bs, v, H_padded, W_padded = 1, 15, 448, 448
    max_valid_points = [100000]
    captions = ["This is a test caption."]
    images_tensor = torch.zeros(bs, v, 3, H_padded, W_padded).to(device) # [B, V, C, H, W]
    images_tensor = rearrange(images_tensor, "b v c h w -> (b v) c h w")
    depths = torch.zeros(bs, v, H_padded, W_padded).to(device) # [B, V, H, W]
    poses = torch.eye(4).repeat(bs, v, 1, 1).to(device) # tensor [B, V, 4, 4]
    intrinsics = torch.eye(4).repeat(bs, v, 1, 1).to(device) # tensor [B, V, 4, 4]
    align_matrix = torch.eye(4).to(device)
    is_train = False

    assert bs == 1
    batched_inputs = []
    for i in range(bs):
        multi_scale_xyz, scannet_pc, original_xyz = get_multiview_xyz(
            shape=(v, H_padded, W_padded),
            size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            depths=[x for x in depths[0]],
            poses=[x for x in poses[0]],
            intrinsics=[x for x in intrinsics[0]],
            is_train=is_train,
            augment_3d=cfg.INPUT.AUGMENT_3D,
            interpolation_method=cfg.MODEL.INTERPOLATION_METHOD,
            mask_valid=cfg.MASK_VALID,
            mean_center=cfg.MEAN_CENTER,
            do_rot_scale=cfg.DO_ROT_SCALE,
            scannet_pc=None,
            align_matrix=align_matrix,
            vil3d=cfg.VIL3D,
            scales=cfg.MULTIVIEW_XYZ_SCALES,
        )
        multiview_data = {
            "multi_scale_xyz": multi_scale_xyz,
        }
        batched_inputs.append(multiview_data)

    _, multiview_data = load_3d_data(
        cfg,
        batched_inputs,
        images_shape=[bs, v, H_padded, W_padded],
        device=images_tensor.device
    )

    multi_scale_xyz = multiview_data["multi_scale_xyz"]
    multi_scale_p2v = multiview_data["multi_scale_p2v"]
    mask_features_xyz = [x.reshape(bs, -1, 3) for x in multi_scale_xyz]
    mask_features_p2v = [x.reshape(bs, -1) for x in multi_scale_p2v]
    scannet_pc = mask_features_xyz[0]
    scannet_p2v = mask_features_p2v[0]

    bs, v, H_padded, W_padded = 1, 15, 448, 448
    shape = (bs, v, H_padded, W_padded)
    # Return all needed variables
    return images_tensor, multiview_data, scannet_pc, scannet_p2v, captions, max_valid_points, shape

def get_saved_data(cfg):
    output_path = Path('ckpts') / 'misc' / 'negations' /'data_sample_scene0329_00.pth'
    data = torch.load(output_path)
    images_tensor = data["images_tensor"]
    multiview_data = data["multiview_data"]
    scannet_pc = data["scannet_pc"]
    scannet_p2v = data["scannet_p2v"]
    captions = data["captions"]
    max_valid_points = data["max_valid_points"]
    shape = data["shape"]
    
    return images_tensor, multiview_data, scannet_pc, scannet_p2v, captions, max_valid_points, shape
@torch.inference_mode()
def fwd(cfg, model):
    use_data = True
    assert not cfg.USE_SEGMENTS
    device = next(model.parameters()).device
    max_bs = 4
    
    # 1. Caricamento Dati
    images_tensor, multiview_data, scannet_pc, scannet_p2v, captions, max_valid_points, shape = \
        get_saved_data(cfg) if use_data else get_dummy_data(cfg, device)
    bs, v, H_padded, W_padded = shape

    if max_bs is not None and max_bs < bs:
        bs = max_bs
    
    multi_scale_xyz = multiview_data["multi_scale_xyz"]
    multi_scale_p2v = multiview_data["multi_scale_p2v"]
    tokenizer = model.mask_decoder.lang_encoder.tokenizer
    
    # 2. Processamento Linguistico
    lang_data = []
    for i in range(bs):
        _dataset_dict = {
            "utterance": captions[i], "target_str": None, "target_id": -1, "anchor_ids": [], "anchors_types": [],
        }
        lang_data.append(process_lang_data(cfg, tokenizer, _dataset_dict))

    # 3. Forward Pass Modello
    mask_features, _ = model.visual_backbone(
        images=images_tensor,
        multi_scale_xyz=multi_scale_xyz,
        multi_scale_p2v=multi_scale_p2v,
        shape=[bs, v, H_padded, W_padded],
        decoder_3d=True,
        actual_decoder_3d=True,
        mesh_pc=scannet_pc,
        mesh_p2v=scannet_p2v
    )

    scannet_pc_mean = scatter_mean(scannet_pc, scannet_p2v, dim=1)
    scannet_p2v_indices = torch.arange(scannet_pc_mean.shape[1], device=scannet_pc.device).unsqueeze(0).repeat(bs, 1)

    outputs = model.mask_decoder(
        mask_features,
        shape=[bs, v],
        mask_features_xyz=scannet_pc_mean,
        mask_features_p2v=scannet_p2v_indices,
        decoder_3d=True,
        captions=captions,
        actual_decoder_3d=True,
        max_valid_points=max_valid_points,
    )

    outputs = get_pred_boxes(outputs, scannet_pc_mean, max_valid_points)
    outputs = get_pred_logits(cfg, lang_data, outputs)
    scores = outputs['pred_scores'][:, :, 0] 

# --- NUOVA LOGICA: SALVATAGGIO DATI GREZZI PER ESPLORAZIONE ---
    img_rgb_flat = rearrange(images_tensor, "(bs v) c h w -> bs (v h w) c", bs=bs, v=v)
    viz_color_all = scatter_mean(img_rgb_flat[:, :scannet_p2v.shape[1], :], scannet_p2v, dim=1)

    for i in range(bs):
        # Prendiamo TUTTI i token, inclusi quelli speciali (utili per il debug)
        tokenized_ids = lang_data[i]["tokenized"]["input_ids"][0]
        tokens_human = tokenizer.convert_ids_to_tokens(tokenized_ids)
        
        layer_to_viz = -1 
        # Estraiamo i tensori core (logits e maschere)
        aux_logits = outputs['aux_outputs'][layer_to_viz]['pred_logits'][i] # [NumQueries, NumTokens]
        aux_masks = outputs['aux_outputs'][layer_to_viz]['pred_masks'][i].sigmoid() # [NumQueries, NumPoints]

        scene_data = {
            "pc": scannet_pc_mean[i].cpu().numpy(),
            "color": (viz_color_all[i].cpu().numpy() * 255).astype(np.uint8),
            "full_caption": captions[i],
            "tokens_human": tokens_human,
            # Salviamo i logit e le maschere grezze
            "raw_logits": aux_logits.cpu().numpy(), 
            "raw_masks": aux_masks.cpu().numpy(),
            "bboxes": box_xyzxyz_to_cxcyczwhd(outputs["pred_boxes"][i]).cpu().numpy()
        }

        # Salviamo un unico file pesante che contiene tutto il potenziale analitico
        output_file = f"outputs/scene_{i}_raw.pth"
        torch.save(scene_data, output_file)
        print(f"   [OK] Dati grezzi salvati in: {output_file}")
        
        # Salviamo il file nella cartella outputs
        # output_file = f"outputs/scene_{i}_data.pth"
        # torch.save(scene_data, output_file)
        # print(f"File salvato: {output_file}")
 

# # --- UNIFIED VISUALIZATION LOOP ---
#     # Pool RGB colors to match the point cloud points (solves the RuntimeError)
#     img_rgb_flat = rearrange(images_tensor, "(bs v) c h w -> bs (v h w) c", bs=bs, v=v)
#     viz_color_all = scatter_mean(img_rgb_flat[:, :scannet_p2v.shape[1], :], scannet_p2v, dim=1)

#     for i in range(bs):
#         # 1. Initialize ONE visualizer for the entire scene
#         v_all = viz.Visualizer()
        
#         # 2. Add the base Room (low alpha so heatmaps are visible)
#         v_all.add_points("00_Base_RGB_Scene", scannet_pc_mean[i].cpu().numpy(), 
#                          colors=(viz_color_all[i].cpu().numpy() * 255).astype(np.uint8), 
#                          alpha=0.2, visible=True, point_size=25)

#         tokenized_ids = lang_data[i]["tokenized"]["input_ids"][0]
#         tokens_human = tokenizer.convert_ids_to_tokens(tokenized_ids)
        
#         layer_to_viz = -1 
#         aux_logits = outputs['aux_outputs'][layer_to_viz]['pred_logits'][i]
#         aux_masks = outputs['aux_outputs'][layer_to_viz]['pred_masks'][i].sigmoid()

#         print(f"\n>>> Generating unified view for: '{captions[i]}'")

#         # 3. Add each word as a toggleable Layer
#         for t_idx, t_name in enumerate(tokens_human):
#             clean_token = t_name.replace('Ġ', '').replace(' ', '')
#             if clean_token in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]', '.', '?', '!']:
#                 continue

#             if t_idx < aux_logits.shape[1]:
#                 query_scores = aux_logits[:, t_idx].sigmoid()
#                 weighted_attn = (aux_masks * query_scores.unsqueeze(-1)).sum(dim=0).cpu().numpy()
                
#                 # Normalize heatmap
#                 attn_norm = (weighted_attn - weighted_attn.min()) / (weighted_attn.max() - weighted_attn.min() + 1e-8)
#                 colors_heatmap = (plt.get_cmap('jet')(attn_norm)[:, :3] * 255).astype(np.uint8)

#                 # Add token layer (hidden by default)
#                 v_all.add_points(f"Token_{t_idx:02d}_{clean_token}", 
#                                  scannet_pc_mean[i].cpu().numpy(), 
#                                  colors=colors_heatmap,
#                                  visible=False,
#                                  point_size=25)

#         # 4. Add Bounding Boxes (Final Model Decisions)
#         max_k = 3
#         top_ids = torch.argsort(scores[i], descending=True)[:max_k]
#         top_bboxes_raw = outputs["pred_boxes"][i][top_ids]
#         top_bboxes_viz = box_xyzxyz_to_cxcyczwhd(top_bboxes_raw).cpu().numpy()
#         pred_colors = get_color(max_k)

#         for b_idx in range(max_k):
#             v_all.add_bounding_box(
#                 f"Pred_Rank_{b_idx}",
#                 position=top_bboxes_viz[b_idx, :3],
#                 size=top_bboxes_viz[b_idx, 3:],
#                 color=np.array(pred_colors[b_idx]),
#                 alpha=0.8,
#                 visible=True
#             )

#         # 5. Save the unified folder
#         save_dir = Path("outputs") / f"unified_scene_{i}_{datetime.now().strftime('%H%M%S')}"
#         save_dir.mkdir(parents=True, exist_ok=True)
#         v_all.save(str(save_dir))
#         print(f"Done! Open the index.html in: {save_dir}")


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    print(f"World size: {comm.get_world_size()}")

    # Currently, each rank will run inference on the same data but this can be modified.
    model = create_ddp_model(
        model,
        broadcast_buffers=False,
        find_unused_parameters=cfg.MULTI_TASK_TRAINING
        or cfg.FIND_UNUSED_PARAMETERS,
    )

    model.eval()
    model.requires_grad_(False)

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    with breakpoint_on_error():
        res = fwd(cfg, model)
        if wandb.run is not None:
            wandb.finish()
        return res

if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "offline"
    args = default_argument_parser().parse_args()
    print(f"Opts: {args.opts}")
    print("Command Line Args:", args)

    _kwargs = dict()
    
    if "launcher=" in args.opts[0]:
        args.opts = args.opts[1:]

    launcher = launch
    _kwargs["args"] = (args,)
    
    # this is needed to prevent memory leak in conv2d layers
    # see: https://github.com/pytorch/pytorch/issues/98688#issuecomment-1869290827
    os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
    launcher(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        **_kwargs
    )