import paddle
import paddle.nn.functional as F
from PIL import Image
import numpy as np
from typing import Iterable

    
def potsdam_class_color_map():
    # RGB
    return {
        0: np.array([255, 255, 255]),
        1: np.array([0, 0, 255]),
        2: np.array([0, 255, 255]),
        3: np.array([0, 255, 0]),
        4: np.array([255, 255, 0]),
        5: np.array([255, 0, 0]),
    }

def six_class_color_map():
    # RGB
    return {
        0: np.array([152, 183, 235]),
        1: np.array([177, 153, 209]),
        2: np.array([145, 194, 159]),
        3: np.array([233, 148, 148]),
        4: np.array([241, 176, 197]),
        5: np.array([239, 172, 137]),
    }
    
def potsdam_class_map():
    return {
        0: 'Impervious surfaces', 1: 'Building', 2: 'Low vegetation',
        3: 'Tree', 4: 'Car', 5: 'Clutter/Background'
    }
    
def label2rgb(label: np.array, class_color_map: dict, size: Iterable):
    # size: (W, H)
    label = label.reshape(size[1], size[0], 1)
    label_colored = np.zeros((size[1], size[0], 3), dtype=np.int32)
    for k, color in class_color_map.items():
        label_colored += color.reshape((1, 1, 3)) * (label==int(k)).astype('int32')
    return label_colored

def rgb2label(label_colored: np.ndarray, class_color_map: dict):
    label = np.ones(label_colored.shape[:2], dtype=np.int32) * -1
    for k, v in class_color_map.items():
        label[np.all(label_colored == v, axis=-1)] = int(k)
    assert np.sum((label==-1).astype('int')) == 0  # check whether all pixels are finished
    return label
    
def get_vis_samples(model, img_paths, gt_paths, size, class_color_map: dict=None):
    # size: (W, H)
    img_tensors, imgs, gts, preds = [], [], [], []
    for i, (img_path, gt_path) in enumerate(zip(img_paths, gt_paths)):
        img, gt = np.array(Image.open(img_path)), np.array(Image.open(gt_path))
        _img = img.transpose((2, 0, 1))
        _img = (_img/255 - 0.5) / 0.5
        _img = paddle.to_tensor(_img, dtype='float32').unsqueeze(0)
        img_tensors.append(_img)
        imgs.append(img)
        gts.append(gt)
    img_tensors = paddle.concat(img_tensors, axis=0)
    _preds = paddle.argmax(F.softmax(model(img_tensors), axis=1), axis=1).numpy()  # (N, H, W)
    
    for i in range(_preds.shape[0]):
        preds.append(
            label2rgb(_preds[i], class_color_map, size) if class_color_map is not None else _preds[i])
        
    return imgs, preds, gts
    
        
    
        