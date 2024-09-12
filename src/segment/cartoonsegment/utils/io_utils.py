
import json, os, sys
import os.path as osp
from typing import List, Union, Tuple, Dict
from pathlib import Path
import cv2
import numpy as np
from imageio import imread, imwrite
import pickle
import pycocotools.mask as maskUtils
from einops import rearrange
from tqdm import tqdm
from PIL import Image
import io
import requests
import traceback
import base64
import time


NP_BOOL_TYPES = (np.bool_, np.bool8)
NP_FLOAT_TYPES = (np.float_, np.float16, np.float32, np.float64)
NP_INT_TYPES = (np.int_, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.ScalarType):
            if isinstance(obj, NP_BOOL_TYPES):
                return bool(obj)
            elif isinstance(obj, NP_FLOAT_TYPES):
                return float(obj)
            elif isinstance(obj, NP_INT_TYPES):
                return int(obj)
        return json.JSONEncoder.default(self, obj)


def json2dict(json_path: str):
    with open(json_path, 'r', encoding='utf8') as f:
        metadata = json.loads(f.read())
    return metadata


def dict2json(adict: dict, json_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(adict, ensure_ascii=False, cls=NumpyEncoder))


def dict2pickle(dumped_path: str, tgt_dict: dict):
    with open(dumped_path, "wb") as f:
        pickle.dump(tgt_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle2dict(pkl_path: str) -> Dict:
    with open(pkl_path, "rb") as f:
        dumped_data = pickle.load(f)
    return dumped_data

def get_all_dirs(root_p: str) -> List[str]: 
    alldir = os.listdir(root_p)
    dirlist = []
    for dirp in alldir:
        dirp = osp.join(root_p, dirp)
        if osp.isdir(dirp):
            dirlist.append(dirp)
    return dirlist


def read_filelist(filelistp: str):
    with open(filelistp, 'r', encoding='utf8') as f:
        lines = f.readlines()
    if len(lines) > 0 and lines[-1].strip() == '':
        lines = lines[:-1]
    return lines


VIDEO_EXTS = {'.flv', '.mp4', '.mkv', '.ts', '.mov', 'mpeg'}
def get_all_videos(video_dir: str, video_exts=VIDEO_EXTS, abs_path=False) -> List[str]:
    filelist = os.listdir(video_dir)
    vlist = []
    for f in filelist:
        if Path(f).suffix in video_exts:
            if abs_path:
                vlist.append(osp.join(video_dir, f))
            else:
                vlist.append(f)
    return vlist


IMG_EXT = {'.bmp', '.jpg', '.png', '.jpeg'}
def find_all_imgs(img_dir, abs_path=False):
    imglist = []
    dir_list = os.listdir(img_dir)
    for filename in dir_list:
        file_suffix = Path(filename).suffix
        if file_suffix.lower() not in IMG_EXT:
            continue
        if abs_path:
            imglist.append(osp.join(img_dir, filename))
        else:
            imglist.append(filename)
    return imglist


def find_all_files_recursive(tgt_dir: Union[List, str], ext, exclude_dirs={}):
    if isinstance(tgt_dir, str):
        tgt_dir = [tgt_dir]

    filelst = []
    for d in tgt_dir:
        for root, _, files in os.walk(d):
            if osp.basename(root) in exclude_dirs:
                continue
            for f in files:
                if Path(f).suffix.lower() in ext:
                    filelst.append(osp.join(root, f))
    
    return filelst


def danbooruid2relpath(id_str: str, file_ext='.jpg'):
    if not isinstance(id_str, str):
        id_str = str(id_str)
    return id_str[-3:].zfill(4) + '/' + id_str  + file_ext


def get_template_histvq(template: np.ndarray) -> Tuple[List[np.ndarray]]:
    len_shape = len(template.shape)
    num_c = 3
    mask = None
    if len_shape == 2:
        num_c = 1
    elif len_shape == 3 and template.shape[-1] == 4:
        mask = np.where(template[..., -1])
        template = template[..., :num_c][mask]

    values, quantiles = [], []
    for ii in range(num_c):
        v, c = np.unique(template[..., ii].ravel(), return_counts=True)
        q = np.cumsum(c).astype(np.float64)
        if len(q) < 1:
            return None, None
        q /= q[-1]
        values.append(v)
        quantiles.append(q)
    return values, quantiles


def inplace_hist_matching(img: np.ndarray, tv: List[np.ndarray], tq: List[np.ndarray]) -> None:
    len_shape = len(img.shape)
    num_c = 3
    mask = None

    tgtimg = img
    if len_shape == 2:
        num_c = 1
    elif len_shape == 3 and img.shape[-1] == 4:
        mask = np.where(img[..., -1])
        tgtimg = img[..., :num_c][mask]

    im_h, im_w = img.shape[:2]
    oldtype = img.dtype
    for ii in range(num_c):
        _, bin_idx, s_counts = np.unique(tgtimg[..., ii].ravel(), return_inverse=True,
                                                return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        if len(s_quantiles) == 0:
            return
        s_quantiles /= s_quantiles[-1]
        interp_t_values = np.interp(s_quantiles, tq[ii], tv[ii]).astype(oldtype)
        if mask is not None:
            img[..., ii][mask] = interp_t_values[bin_idx]
        else:
            img[..., ii] = interp_t_values[bin_idx].reshape((im_h, im_w))
            # try:
            #     img[..., ii] = interp_t_values[bin_idx].reshape((im_h, im_w))
            # except:
            #     LOGGER.error('##################### sth goes wrong')
            #     cv2.imshow('img', img)
            #     cv2.waitKey(0)


def fgbg_hist_matching(fg_list: List, bg: np.ndarray, min_tq_num=128):
    btv, btq = get_template_histvq(bg)
    ftv, ftq = get_template_histvq(fg_list[0]['image'])
    num_fg = len(fg_list)
    idx_matched = -1
    if num_fg > 1:
        _ftv, _ftq = get_template_histvq(fg_list[0]['image'])
        if _ftq is not None and ftq is not None:
            if len(_ftq[0]) > len(ftq[0]):
                idx_matched = num_fg - 1
                ftv, ftq = _ftv, _ftq
            else:
                idx_matched = 0

    if btq is not None and ftq is not None:
        if len(btq[0]) > len(ftq[0]):
            tv, tq = btv, btq
            idx_matched = -1
        else:
            tv, tq = ftv, ftq
            if len(tq[0]) > min_tq_num:
                inplace_hist_matching(bg, tv, tq)
        
        if len(tq[0]) > min_tq_num:
            for ii, fg_dict in enumerate(fg_list):
                fg = fg_dict['image']
                if ii != idx_matched and len(tq[0]) > min_tq_num:
                    inplace_hist_matching(fg, tv, tq)


def imread_nogrey_rgb(imp: str) -> np.ndarray:
    img: np.ndarray = imread(imp)
    c = 1
    if len(img.shape) == 3:
        c = img.shape[-1]
    if c == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if c == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def square_pad_resize(img: np.ndarray, tgt_size: int, pad_value: Tuple = (114, 114, 114)):
    h, w = img.shape[:2]
    pad_h, pad_w = 0, 0
    
    # make square image
    if w < h:
        pad_w = h - w
        w += pad_w
    elif h < w:
        pad_h = w - h
        h += pad_h

    pad_size = tgt_size - h
    if pad_size > 0:
        pad_h += pad_size
        pad_w += pad_size

    if pad_h > 0 or pad_w > 0:    
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)

    down_scale_ratio = tgt_size / img.shape[0]
    assert down_scale_ratio <= 1
    if down_scale_ratio < 1:
        img = cv2.resize(img, (tgt_size, tgt_size), interpolation=cv2.INTER_AREA)

    return img, down_scale_ratio, pad_h, pad_w


def scaledown_maxsize(img: np.ndarray, max_size: int, divisior: int = None):
    
    im_h, im_w = img.shape[:2]
    ori_h, ori_w = img.shape[:2]
    resize_ratio = max_size / max(im_h, im_w)
    if resize_ratio < 1:
        if im_h > im_w:
            im_h = max_size
            im_w = max(1, int(round(im_w * resize_ratio)))
        
        else:
            im_w = max_size
            im_h = max(1, int(round(im_h * resize_ratio)))
    if divisior is not None:
        im_w = int(np.ceil(im_w / divisior) * divisior)
        im_h = int(np.ceil(im_h / divisior) * divisior)

    if im_w != ori_w or im_h != ori_h:
       img = cv2.resize(img, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
       
    return img


def resize_pad(img: np.ndarray, tgt_size: int, pad_value: Tuple = (0, 0, 0)):
    # downscale to tgt_size and pad to square
    img = scaledown_maxsize(img, tgt_size)
    padl, padr, padt, padb = 0, 0, 0, 0
    h, w = img.shape[:2]
    # padt = (tgt_size - h) // 2
    # padb = tgt_size - h - padt
    # padl = (tgt_size - w) // 2
    # padr = tgt_size - w - padl
    padb = tgt_size - h
    padr = tgt_size - w

    if padt + padb + padl + padr > 0:
        img = cv2.copyMakeBorder(img, padt, padb, padl, padr, cv2.BORDER_CONSTANT, value=pad_value)

    return img, (padt, padb, padl, padr)


def resize_pad2divisior(img: np.ndarray, tgt_size: int, divisior: int = 64, pad_value: Tuple = (0, 0, 0)):
    img = scaledown_maxsize(img, tgt_size)
    img, (pad_h, pad_w) = pad2divisior(img, divisior, pad_value)
    return img, (pad_h, pad_w)


def img2grey(img: Union[np.ndarray, str], is_rgb: bool = False) -> np.ndarray:
    if isinstance(img, np.ndarray):
        if len(img.shape) == 3:
            if img.shape[-1] != 1:
                if is_rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = img[..., 0]
        return img
    elif isinstance(img, str):
        return cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    else:
        raise NotImplementedError
    

def pad2divisior(img: np.ndarray, divisior: int, value = (0, 0, 0)) -> np.ndarray:
    im_h, im_w = img.shape[:2]
    pad_h = int(np.ceil(im_h / divisior)) * divisior - im_h
    pad_w = int(np.ceil(im_w / divisior)) * divisior - im_w
    if pad_h != 0 or pad_w != 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, value=value, borderType=cv2.BORDER_CONSTANT)
    return img, (pad_h, pad_w)


def mask2rle(mask: np.ndarray, decode_for_json: bool = True) -> Dict:
    mask_rle = maskUtils.encode(np.array(
                        mask[..., np.newaxis] > 0, order='F',
                        dtype='uint8'))[0]
    if decode_for_json:
        mask_rle['counts'] = mask_rle['counts'].decode()
    return mask_rle


def bbox2xyxy(box) -> Tuple[int]:
    x1, y1 = box[0], box[1]
    return x1, y1, x1+box[2], y1+box[3]


def bbox_overlap_area(abox, boxb) -> int:
    ax1, ay1, ax2, ay2 = bbox2xyxy(abox)
    bx1, by1, bx2, by2 = bbox2xyxy(boxb)
    
    ix = min(ax2, bx2) - max(ax1, bx1)
    iy = min(ay2, by2) - max(ay1, by1)
    
    if ix > 0 and iy > 0:
        return ix * iy
    else:
        return 0


def bbox_overlap_xy(abox, boxb) -> Tuple[int]:
    ax1, ay1, ax2, ay2 = bbox2xyxy(abox)
    bx1, by1, bx2, by2 = bbox2xyxy(boxb)
    
    ix = min(ax2, bx2) - max(ax1, bx1)
    iy = min(ay2, by2) - max(ay1, by1)
    
    return ix, iy


def xyxy_overlap_area(axyxy, bxyxy) -> int:
    ax1, ay1, ax2, ay2 = axyxy
    bx1, by1, bx2, by2 = bxyxy
    
    ix = min(ax2, bx2) - max(ax1, bx1)
    iy = min(ay2, by2) - max(ay1, by1)
    
    if ix > 0 and iy > 0:
        return ix * iy
    else:
        return 0


DIRNAME2TAG = {'rezero': 're:zero'}
def dirname2charactername(dirname, start=6):
    cname = dirname[start:]
    for k, v in DIRNAME2TAG.items():
        cname = cname.replace(k, v)
    return cname


def imglist2grid(imglist: np.ndarray, grid_size: int = 384, col=None) -> np.ndarray:
    sqimlist = []
    for img in imglist:
        sqimlist.append(square_pad_resize(img, grid_size)[0])

    nimg = len(imglist)
    if nimg == 0:
        return None
    padn = 0
    if col is None:
        if nimg > 5:
            row = int(np.round(np.sqrt(nimg)))
            col = int(np.ceil(nimg / row))
        else:
            col = nimg

    padn = int(np.ceil(nimg / col) * col) - nimg
    if padn != 0:
        padimg = np.zeros_like(sqimlist[0])
        for _ in range(padn):
            sqimlist.append(padimg)
    
    return rearrange(sqimlist, '(row col) h w c -> (row h) (col w) c', col=col)

def write_jsonlines(filep: str, dict_lst: List[str], progress_bar: bool = True):
    with open(filep, 'w') as out:
        if progress_bar:
            lst = tqdm(dict_lst)
        else:
            lst = dict_lst
        for ddict in lst:
            jout = json.dumps(ddict) + '\n'
            out.write(jout)

def read_jsonlines(filep: str):
    with open(filep, 'r', encoding='utf8') as f:
        result = [json.loads(jline) for jline in f.read().splitlines()]
    return result


def _b64encode(x: bytes) -> str:
    return base64.b64encode(x).decode("utf-8")


def img2b64(img):
    """
    Convert a PIL image to a base64-encoded string.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    return _b64encode(buffered.getvalue())


def save_encoded_image(b64_image: str, output_path: str):
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))

def submit_request(url, data, exist_on_exception=True, auth=None, wait_time = 30):
    response = None
    try:
        while True:
            try:
                response = requests.post(url, data=data, auth=auth)
                response.raise_for_status()
                break
            except Exception as e:
                if wait_time > 0:
                    print(traceback.format_exc(), file=sys.stderr)
                    print(f'sleep {wait_time} sec...')
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        if response is not None:
            print('response content: ' + response.text)
        if exist_on_exception:
            exit()
    return response


# def resize_image(input_image, resolution):
#     H, W = input_image.shape[:2]
#     k = float(min(resolution)) / min(H, W)
#     img = cv2.resize(input_image, resolution, interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
#     return img
