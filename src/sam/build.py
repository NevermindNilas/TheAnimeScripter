"""Builds the SAM 2.1 Hiera Base+ video predictor without hydra.

The values are those of upstream's ``configs/sam2.1/sam2.1_hiera_b+.yaml`` plus the
overrides ``build_sam2_video_predictor`` applies for video. Both weight sets TAS
offers (the Meta release and the anime finetune) use this architecture.
"""

import glob
import logging
import os
import re

import torch

# Both files are mirrored byte for byte on the TAS model host (Apache-2.0).
# Upstream: huggingface.co/Zarxrax/SAM2Anime (anime, sha256 2e9802ca...) and
# dl.fbaipublicfiles.com/segment_anything_2/092824/ (base, sha256 a2345aed...).
MODELS = {
    "anime": "sam2.1_anime_v1.pt",
    "base": "sam2.1_hiera_base_plus.pt",
}


def weightPath(model: str) -> str:
    """The checkpoint for ``model``, downloaded the first time it is needed."""
    from src.model.download import downloadAndLog
    from src.model.registry import TASURL, weightsDir

    filename = MODELS[model]
    folder = os.path.join(weightsDir, "sam2")
    os.makedirs(folder, exist_ok=True)
    removeAbandonedDownloads(folder, filename)
    return downloadAndLog(f"sam2-{model}", filename, TASURL + filename, folder)


def removeAbandonedDownloads(folder: str, filename: str) -> None:
    """Delete partial downloads of ``filename`` whose process is gone.

    A download writes ``TEMP/<filename>.<pid>.part``. Killing the session while it
    runs leaves that behind under a name no later run will reuse. A file whose
    process is alive stays, since another run may be writing it. Windows may
    still refuse to delete one, which is no reason to fail the open.
    """
    import psutil

    pattern = re.compile(re.escape(filename) + r"\.(\d+)\.part")
    temporary = os.path.join(folder, "TEMP")
    for path in glob.glob(os.path.join(glob.escape(temporary), "*.part")):
        match = pattern.fullmatch(os.path.basename(path))
        if match is None or psutil.pid_exists(int(match.group(1))):
            continue
        try:
            os.remove(path)
        except OSError as error:
            logging.warning(f"Could not remove {path}: {error}")


def _rope(**extra):
    from src.sam.sam2.modeling.sam.transformer import RoPEAttention

    return RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=[64, 64],
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        **extra,
    )


def buildPredictor(checkpoint: str, device):
    from src.sam.sam2.modeling.backbones.hieradet import Hiera
    from src.sam.sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
    from src.sam.sam2.modeling.memory_attention import (
        MemoryAttention,
        MemoryAttentionLayer,
    )
    from src.sam.sam2.modeling.memory_encoder import (
        CXBlock,
        Fuser,
        MaskDownSampler,
        MemoryEncoder,
    )
    from src.sam.sam2.modeling.position_encoding import PositionEmbeddingSine
    from src.sam.sam2.sam2_video_predictor import SAM2VideoPredictor

    model = SAM2VideoPredictor(
        image_encoder=ImageEncoder(
            scalp=1,
            trunk=Hiera(embed_dim=112, num_heads=2),
            neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(
                    num_pos_feats=256, normalize=True, scale=None, temperature=10000
                ),
                d_model=256,
                backbone_channel_list=[896, 448, 224, 112],
                fpn_top_down_levels=[2, 3],
                fpn_interp_model="nearest",
            ),
        ),
        memory_attention=MemoryAttention(
            d_model=256,
            pos_enc_at_input=True,
            layer=MemoryAttentionLayer(
                activation="relu",
                dim_feedforward=2048,
                dropout=0.1,
                pos_enc_at_attn=False,
                self_attention=_rope(),
                d_model=256,
                pos_enc_at_cross_attn_keys=True,
                pos_enc_at_cross_attn_queries=False,
                cross_attention=_rope(rope_k_repeat=True, kv_in_dim=64),
            ),
            num_layers=4,
        ),
        memory_encoder=MemoryEncoder(
            out_dim=64,
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=64, normalize=True, scale=None, temperature=10000
            ),
            mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
            fuser=Fuser(
                layer=CXBlock(
                    dim=256,
                    kernel_size=7,
                    padding=3,
                    layer_scale_init_value=1e-6,
                    use_dwconv=True,
                ),
                num_layers=2,
            ),
        ),
        num_maskmem=7,
        image_size=1024,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        no_obj_embed_spatial=True,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
        compile_image_encoder=False,
        # What upstream adds for video.
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        binarize_mask_from_pts_for_mem_enc=True,
        fill_hole_area=8,
        # A click on a frame the track already reached has to become a conditioning
        # frame. Otherwise it only changes the preview: the next propagate replaces
        # that frame's output with one that ignores the click.
        add_all_frames_to_correct_as_cond=True,
    )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state)
    logging.info(f"Loaded SAM 2 checkpoint: {checkpoint}")
    return model.to(device).eval()
