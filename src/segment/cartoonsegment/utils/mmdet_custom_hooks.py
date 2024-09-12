from mmengine.fileio import FileClient
from mmengine.dist import master_only
from einops import rearrange
import torch
import mmcv
import numpy as np
import os.path as osp
import cv2
from typing import Optional, Sequence
import torch.nn as nn
from mmdet.apis import inference_detector
from mmcv.transforms import Compose
from mmdet.engine import DetVisualizationHook
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample

from utils.io_utils import find_all_imgs, square_pad_resize, imglist2grid

def inference_detector(
    model: nn.Module,
    imgs,
    test_pipeline
):

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    if len(imgs) == 0:
        return []

    test_pipeline = test_pipeline.copy()
    if isinstance(imgs[0], np.ndarray):
        # Calling this method across libraries will result
        # in module unregistered error if not prefixed with mmdet.
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

    test_pipeline = Compose(test_pipeline)

    result_list = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)
        # build the data pipeline
        data_ = test_pipeline(data_)

        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        # forward the model
        with torch.no_grad():
            results = model.test_step(data_)[0]

        result_list.append(results)

    if not is_batch:
        return result_list[0]
    else:
        return result_list


@HOOKS.register_module()
class InstanceSegVisualizationHook(DetVisualizationHook):

    def __init__(self, visualize_samples: str = '', 
                 read_rgb: bool = False,
                 draw: bool = False,
                 interval: int = 50,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk')):
        super().__init__(draw, interval, score_thr, show, wait_time, test_out_dir, file_client_args)
        self.vis_samples = []

        if osp.exists(visualize_samples):
            self.channel_order = channel_order = 'rgb' if read_rgb else 'bgr' 
            samples = find_all_imgs(visualize_samples, abs_path=True)
            for imgp in samples:
                img = mmcv.imread(imgp, channel_order=channel_order)
                img, _, _, _ = square_pad_resize(img, 640)
                self.vis_samples.append(img)

    def before_val(self, runner) -> None:
        total_curr_iter = runner.iter
        self._visualize_data(total_curr_iter, runner)
        return super().before_val(runner)

    # def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
    #                    outputs: Sequence[DetDataSample]) -> None:
    #     """Run after every ``self.interval`` validation iterations.

    #     Args:
    #         runner (:obj:`Runner`): The runner of the validation process.
    #         batch_idx (int): The index of the current batch in the val loop.
    #         data_batch (dict): Data from dataloader.
    #         outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
    #             that contain annotations and predictions.
    #     """
    #     # if self.draw is False:
    #     #     return

    #     if self.file_client is None:
    #         self.file_client = FileClient(**self.file_client_args)


    #     # There is no guarantee that the same batch of images
    #     # is visualized for each evaluation.
    #     total_curr_iter = runner.iter + batch_idx

    #     # # Visualize only the first data
    #     # img_path = outputs[0].img_path
    #     # img_bytes = self.file_client.get(img_path)
    #     # img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
    #     if total_curr_iter % self.interval == 0 and self.vis_samples:
    #         self._visualize_data(total_curr_iter, runner)


    @master_only
    def _visualize_data(self, total_curr_iter, runner):

        tgt_size = 384

        runner.model.eval()
        outputs = inference_detector(runner.model, self.vis_samples, test_pipeline=runner.cfg.test_pipeline)
        vis_results = []
        for img, output in zip(self.vis_samples, outputs):
            vis_img = self.add_datasample(
                    'val_img',
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    pred_score_thr=self.score_thr,
                    draw_gt=False,
                    step=total_curr_iter)
            vis_results.append(cv2.resize(vis_img, (tgt_size, tgt_size), interpolation=cv2.INTER_AREA))

        drawn_img = imglist2grid(vis_results, tgt_size)
        if drawn_img is None:
            return
        drawn_img = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB)
        visualizer = self._visualizer
        visualizer.set_image(drawn_img)
        visualizer.add_image('val_img', drawn_img, total_curr_iter)


    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> np.ndarray:
        image = image.clip(0, 255).astype(np.uint8)
        visualizer = self._visualizer
        classes = visualizer.dataset_meta.get('classes', None)
        palette = visualizer.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            gt_img_data = image
            if 'gt_instances' in data_sample:
                gt_img_data = visualizer._draw_instances(image,
                                                   data_sample.gt_instances,
                                                   classes, palette)

            if 'gt_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                gt_img_data = visualizer._draw_panoptic_seg(
                    gt_img_data, data_sample.gt_panoptic_seg, classes)

        if draw_pred and data_sample is not None:
            pred_img_data = image
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr]
                pred_img_data = visualizer._draw_instances(image, pred_instances,
                                                     classes, palette)
            if 'pred_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                pred_img_data = visualizer._draw_panoptic_seg(
                    pred_img_data, data_sample.pred_panoptic_seg.numpy(),
                    classes)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        return drawn_img
