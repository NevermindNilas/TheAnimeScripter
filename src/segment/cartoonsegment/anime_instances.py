import numpy as np
from typing import List, Union
import torch
from utils.constants import COLOR_PALETTE
from utils.constants import get_color
import cv2


def tags2multilines(tags: Union[str, List], lw, tf, max_width):
    if isinstance(tags, str):
        taglist = tags.split(" ")
    else:
        taglist = tags

    sz = cv2.getTextSize(" ", 0, lw / 3, tf)
    line_height = sz[0][1]
    line_width = 0
    if len(taglist) > 0:
        lines = [taglist[0]]
        if len(taglist) > 1:
            for t in taglist[1:]:
                textl = len(t) * line_height
                if line_width + line_height + textl > max_width:
                    lines.append(t)
                    line_width = 0
                else:
                    line_width = line_width + line_height + textl
                    lines[-1] = lines[-1] + " " + t
    return lines, line_height


class AnimeInstances:
    def __init__(
        self,
        masks: Union[np.ndarray, torch.Tensor] = None,
        bboxes: Union[np.ndarray, torch.Tensor] = None,
        scores: Union[np.ndarray, torch.Tensor] = None,
        tags: List[str] = None,
        character_tags: List[str] = None,
    ) -> None:
        self.masks = masks
        self.tags = tags
        self.bboxes = bboxes

        if scores is None:
            scores = [1.0] * len(self)
            if self.is_numpy:
                scores = np.array(scores)
            elif self.is_tensor:
                scores = torch.tensor(scores)

        self.scores = scores

        if tags is None:
            self.tags = [""] * len(self)
            self.character_tags = [""] * len(self)
        else:
            self.tags = tags
            self.character_tags = character_tags

    @property
    def is_cuda(self):
        if isinstance(self.masks, torch.Tensor) and self.masks.is_cuda:
            return True
        else:
            return False

    @property
    def is_tensor(self):
        if self.is_empty:
            return False
        else:
            return isinstance(self.masks, torch.Tensor)

    @property
    def is_numpy(self):
        if self.is_empty:
            return True
        else:
            return isinstance(self.masks, np.ndarray)

    @property
    def is_empty(self):
        return self.masks is None or len(self.masks) == 0

    def remove_duplicated(self):
        num_masks = len(self)
        if num_masks < 2:
            return

        need_cvt = False
        if self.is_numpy:
            need_cvt = True
            self.to_tensor()

        mask_areas = torch.Tensor([mask.sum() for mask in self.masks])
        sids = torch.argsort(mask_areas, descending=True)
        sids = sids.cpu().numpy().tolist()
        mask_areas = mask_areas[sids]
        masks = self.masks[sids]
        bboxes = self.bboxes[sids]
        tags = [self.tags[sid] for sid in sids]
        scores = self.scores[sids]

        canvas = masks[0]

        valid_ids: List = np.arange(num_masks).tolist()
        for ii, mask in enumerate(masks[1:]):
            mask_id = ii + 1
            canvas_and = torch.bitwise_and(canvas, mask)

            and_area = canvas_and.sum()
            mask_area = mask_areas[mask_id]

            if and_area / mask_area > 0.8:
                valid_ids.remove(mask_id)
            elif mask_id != num_masks - 1:
                canvas = torch.bitwise_or(canvas, mask)

        sids = valid_ids
        self.masks = masks[sids]
        self.bboxes = bboxes[sids]
        self.tags = [tags[sid] for sid in sids]
        self.scores = scores[sids]

        if need_cvt:
            self.to_numpy()

        # sids =

    def draw_instances(
        self,
        img: np.ndarray,
        draw_bbox: bool = True,
        draw_ins_mask: bool = True,
        draw_ins_contour: bool = True,
        draw_tags: bool = False,
        draw_indices: List = None,
        mask_alpha: float = 0.4,
    ):
        mask_alpha = 0.75

        drawed = img.copy()

        if self.is_empty:
            return drawed

        im_h, im_w = img.shape[:2]

        mask_shape = self.masks[0].shape
        if mask_shape[0] != im_h or mask_shape[1] != im_w:
            drawed = cv2.resize(
                drawed, (mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_AREA
            )
            im_h, im_w = mask_shape[0], mask_shape[1]

        if draw_indices is None:
            draw_indices = list(range(len(self)))
        ins_dict = {
            "mask": [],
            "tags": [],
            "score": [],
            "bbox": [],
            "character_tags": [],
        }
        colors = []
        for idx in draw_indices:
            ins = self.get_instance(idx, out_type="numpy")
            for key, data in ins.items():
                ins_dict[key].append(data)
            colors.append(get_color(idx))

        if draw_bbox:
            lw = max(round(sum(drawed.shape) / 2 * 0.003), 2)
            for color, bbox in zip(colors, ins_dict["bbox"]):
                p1, p2 = (
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])),
                )
                cv2.rectangle(drawed, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

        if draw_ins_mask:
            drawed = drawed.astype(np.float32)
            for color, mask in zip(colors, ins_dict["mask"]):
                p = mask.astype(np.float32)
                blend_mask = np.full((im_h, im_w, 3), color, dtype=np.float32)
                alpha_msk = (mask_alpha * p)[..., None]
                alpha_ori = 1 - alpha_msk
                drawed = drawed * alpha_ori + alpha_msk * blend_mask
            drawed = drawed.astype(np.uint8)

        if draw_tags:
            lw = max(round(sum(drawed.shape) / 2 * 0.002), 2)
            tf = max(lw - 1, 1)
            for color, tags, bbox in zip(colors, ins_dict["tags"], ins_dict["bbox"]):
                if not tags:
                    continue
                lines, line_height = tags2multilines(tags, lw, tf, bbox[2])
                for ii, l in enumerate(lines):
                    xy = (bbox[0], bbox[1] + line_height + int(line_height * 1.2 * ii))
                    cv2.putText(
                        drawed,
                        l,
                        xy,
                        0,
                        lw / 3,
                        color,
                        thickness=tf,
                        lineType=cv2.LINE_AA,
                    )

        # cv2.imshow('canvas', drawed)
        # cv2.waitKey(0)
        return drawed

    def cuda(self):
        if self.is_empty:
            return self
        self.to_tensor(device="cuda")
        return self

    def cpu(self):
        if not self.is_tensor or not self.is_cuda:
            return self
        self.masks = self.masks.cpu()
        self.scores = self.scores.cpu()
        self.bboxes = self.bboxes.cpu()
        return self

    def to_tensor(self, device: str = "cpu"):
        if self.is_empty:
            return self
        elif self.is_tensor and self.masks.device == device:
            return self
        self.masks = torch.from_numpy(self.masks).to(device)
        self.bboxes = torch.from_numpy(self.bboxes).to(device)
        self.scores = torch.from_numpy(self.scores).to(device)
        return self

    def to_numpy(self):
        if self.is_numpy:
            return self
        if self.is_cuda:
            self.masks = self.masks.cpu().numpy()
            self.scores = self.scores.cpu().numpy()
            self.bboxes = self.bboxes.cpu().numpy()
        else:
            self.masks = self.masks.numpy()
            self.scores = self.scores.numpy()
            self.bboxes = self.bboxes.numpy()
        return self

    def get_instance(self, ins_idx: int, out_type: str = None, device: str = None):
        mask = self.masks[ins_idx]
        tags = self.tags[ins_idx]
        character_tags = self.character_tags[ins_idx]
        bbox = self.bboxes[ins_idx]
        score = self.scores[ins_idx]
        if out_type is not None:
            if out_type == "numpy" and not self.is_numpy:
                mask = mask.cpu().numpy()
                bbox = bbox.cpu().numpy()
                score = score.cpu().numpy()
            if out_type == "tensor" and not self.is_tensor:
                mask = torch.from_numpy(mask)
                bbox = torch.from_numpy(bbox)
                score = torch.from_numpy(score)
            if (
                isinstance(mask, torch.Tensor)
                and device is not None
                and mask.device != device
            ):
                mask = mask.to(device)
                bbox = bbox.to(device)
                score = score.to(device)

        return {
            "mask": mask,
            "tags": tags,
            "character_tags": character_tags,
            "bbox": bbox,
            "score": score,
        }

    def __len__(self):
        if self.is_empty:
            return 0
        else:
            return len(self.masks)

    def resize(self, h, w, mode="area"):
        if self.is_empty:
            return
        if self.is_tensor:
            masks = self.masks.to(torch.float).unsqueeze(1)
            oh, ow = masks.shape[2], masks.shape[3]
            hs, ws = h / oh, w / ow
            bboxes = self.bboxes.float()
            bboxes[:, ::2] *= hs
            bboxes[:, 1::2] *= ws
            self.bboxes = torch.round(bboxes).int()
            masks = torch.nn.functional.interpolate(masks, (h, w), mode=mode)
            self.masks = masks.squeeze(1) > 0.3

    def compose_masks(self, output_type=None):
        if self.is_empty:
            return None
        else:
            mask = self.masks[0]
            if len(self.masks) > 1:
                for m in self.masks[1:]:
                    if self.is_numpy:
                        mask = np.logical_or(mask, m)
                    else:
                        mask = torch.logical_or(mask, m)
            if output_type is not None:
                if output_type == "numpy" and not self.is_numpy:
                    mask = mask.cpu().numpy()
                if output_type == "tensor" and not self.is_tensor:
                    mask = torch.from_numpy(mask)
            return mask
