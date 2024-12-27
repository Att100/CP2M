import os
import cv2
import paddle
from PIL import Image
import albumentations as A
import numpy as np
import random
from paddle.io import Dataset

from utils.vis import rgb2label, potsdam_class_color_map


class Potsdam(Dataset):
    SIZE = (512, 512)
    
    def __init__(
        self, img_path="./dataset/Potsdam/images2", gt_path="./dataset/Potsdam/labels2", 
        dtsplit='train', p_mosaic=0.8, p_cpm=0.5, col=6, row=6):
        
        super().__init__()
        
        self.dtsplit = dtsplit
        self.col, self.row = col, row
        self.img_path, self.gt_path = img_path, gt_path
        
        self.imgs, self.labels = self._get_paths()
        self.index = [i for i in range(len(self.imgs))]
        self.p_mosaic = p_mosaic
        self.p_cpm = p_cpm
        
        if self.p_mosaic > 0:
            self.mosaic_transform = MosaicAugmentation((self.SIZE[0]//2, self.SIZE[1]//2))
        else: self.mosaic_transform = None
        
        if self.p_cpm > 0:
            self.cpm_transform = ClustedPatchMixAugmentation(self.SIZE, (self.SIZE[0]//2, self.SIZE[1]//2))
        else: self.cpm_transform = None
        
        
    def __getitem__(self, idx):
        img, label = self._make_sample(idx)
        img = img.transpose((2, 0, 1))
        img = (img/255 - 0.5) / 0.5
        img, label = paddle.to_tensor(img, dtype='float32'), paddle.to_tensor(label, dtype='int64')
        return img, label
    
    def __len__(self):
        return len(self.imgs)
    
    def _make_sample(self, idx):
        if random.random() < self.p_mosaic:
            sampled_index = random.sample(self.index, 4)
            sub_imgs, sub_masks = [], []
            for i in sampled_index:
                sub_imgs.append(np.array(Image.open(self.imgs[i])))
                sub_masks.append(np.array(Image.open(self.labels[i])))
            img, label = self.mosaic_transform(images=sub_imgs, masks=sub_masks)
        else:
            img, label = Image.open(self.imgs[idx]), Image.open(self.labels[idx])
            img, label = np.array(img), np.array(label)
            if len(label.shape)==3 and label.shape[-1]==3:
                label = rgb2label(label, potsdam_class_color_map())

        if random.random() < self.p_cpm:
            sampled_index = random.randint(0, len(self.index)-1)
            img2 = np.array(Image.open(self.imgs[sampled_index]))
            label2 = np.array(Image.open(self.labels[sampled_index]))
            _img, _label = self.cpm_transform(img, label, img2, label2)
            if _img is not None:
                return _img, _label
            else:
                sampled_index = random.randint(0, len(self.index)-1)
                img2 = np.array(Image.open(self.imgs[sampled_index]))
                label2 = np.array(Image.open(self.labels[sampled_index]))
                _img, _label = self.cpm_transform(img, label, img2, label2)
                if _img is not None:
                    return _img, _label
            
        return img, label
    
    def _get_paths(self):
        if self.dtsplit == 'train':
            names = self._get_train_file_head()
        else:
            names = self._get_test_file_head()
            
        imgs, labels = [], []
        for name in names:
            for i in range(self.col):
                for j in range(self.row):
                    fname = f'{name}_{i+1}_{j+1}'
                    imgs.append(os.path.join(self.img_path, fname+"_RGB.tif"))
                    labels.append(os.path.join(self.gt_path, fname+"_label.tif"))
                    
        return imgs, labels
        
    def _get_train_file_head(self):
        return [
            'top_potsdam_2_10', 'top_potsdam_2_11', 'top_potsdam_2_12', 'top_potsdam_3_10',
            'top_potsdam_3_11', 'top_potsdam_3_12', 'top_potsdam_4_10', 'top_potsdam_4_11',
            'top_potsdam_4_12', 'top_potsdam_5_10', 'top_potsdam_5_11', 'top_potsdam_5_12', 
            'top_potsdam_6_7', 'top_potsdam_6_8', 'top_potsdam_6_9', 'top_potsdam_6_10',
            'top_potsdam_6_11', 'top_potsdam_6_12', 'top_potsdam_7_7', 'top_potsdam_7_8',
            'top_potsdam_7_9', 'top_potsdam_7_10', 'top_potsdam_7_11', 'top_potsdam_7_12',
        ]
    
    def _get_test_file_head(self):
        return [
            'top_potsdam_2_13', 'top_potsdam_2_14', 'top_potsdam_3_13', 'top_potsdam_3_14',
            'top_potsdam_4_13', 'top_potsdam_4_14', 'top_potsdam_4_15', 'top_potsdam_5_13',
            'top_potsdam_5_14', 'top_potsdam_5_15', 'top_potsdam_6_13', 'top_potsdam_6_14', 
            'top_potsdam_6_15', 'top_potsdam_7_13'
        ]


class MosaicAugmentation:
    def __init__(self, sub_img_size=(512, 512), rotation=(-90, 90), p_rot=0.8, p_hf=0.5, p_vf=0.5) -> None:
        self.sub_img_size = sub_img_size  # H, W
        self.sub_img_transforms = A.Compose([
            A.HorizontalFlip(p=p_hf),
            A.VerticalFlip(p=p_vf),
            A.Rotate(limit=rotation, crop_border=True, p=p_rot),
            A.RandomCrop(sub_img_size[0], sub_img_size[1])
        ])
        
    def __call__(self, images, masks):
        # images: [(H, W, 3), ...]
        # labels: [(H, W, 3), ...] or [(H, W), ...]
        sub_imgs, sub_masks = [], []
        for img, mask in zip(images, masks):
            _img, _mask = self._crop_sub_img(img, mask)
            sub_imgs.append(_img)
            sub_masks.append(_mask)
        return self._combine_sub_imgs_masks(sub_imgs, sub_masks)
        
    def _crop_sub_img(self, img: np.ndarray, label: np.ndarray):
        transformed = self.sub_img_transforms(image=img, mask=label)
        return transformed['image'], transformed['mask']
    
    def _combine_sub_imgs_masks(self, sub_imgs, sub_masks):
        # combine imgs
        img_out = np.zeros((2*self.sub_img_size[0], 2*self.sub_img_size[1], 3), dtype='int32')
        img_out[:self.sub_img_size[0], :self.sub_img_size[1], :] = sub_imgs[0]  # top-left
        img_out[:self.sub_img_size[0], self.sub_img_size[1]:, :] = sub_imgs[1]  # top-right
        img_out[self.sub_img_size[0]:, :self.sub_img_size[1], :] = sub_imgs[2]  # bottom-left
        img_out[self.sub_img_size[0]:, self.sub_img_size[1]:, :] = sub_imgs[3]  # bottom-right
        
        # combine masks
        if len(sub_masks[0].shape)==3 and sub_masks[0].shape[-1]==3:
            mask_out = np.zeros((2*self.sub_img_size[0], 2*self.sub_img_size[1], 3), dtype='int32')
            mask_out[:self.sub_img_size[0], :self.sub_img_size[1], :] = sub_masks[0]  # top-left
            mask_out[:self.sub_img_size[0], self.sub_img_size[1]:, :] = sub_masks[1]  # top-right
            mask_out[self.sub_img_size[0]:, :self.sub_img_size[1], :] = sub_masks[2]  # bottom-left
            mask_out[self.sub_img_size[0]:, self.sub_img_size[1]:, :] = sub_masks[3]  # bottom-right
        else:
            mask_out = np.zeros((2*self.sub_img_size[0], 2*self.sub_img_size[1]), dtype='int32')
            mask_out[:self.sub_img_size[0], :self.sub_img_size[1]] = sub_masks[0]  # top-left
            mask_out[:self.sub_img_size[0], self.sub_img_size[1]:] = sub_masks[1]  # top-right
            mask_out[self.sub_img_size[0]:, :self.sub_img_size[1]] = sub_masks[2]  # bottom-left
            mask_out[self.sub_img_size[0]:, self.sub_img_size[1]:] = sub_masks[3]  # bottom-right
            
        return img_out, mask_out


class ClustedPatchMixAugmentation:
    def __init__(
        self, size=(1024, 1024), crop_size=(512, 512), rotation=(-90, 90), 
        p_rot=0.8, p_hf=0.5, p_vf=0.5, include_classes=[3, 4, 5], n_patch=10) -> None:
        self.size = size
        self.crop_size = crop_size
        self.include_classes = include_classes
        self.npatch = n_patch
        
        self.cpm_transforms = A.Compose([
            A.HorizontalFlip(p=p_hf),
            A.VerticalFlip(p=p_vf),
            A.Rotate(limit=rotation, crop_border=True, p=p_rot),
            A.RandomCrop(crop_size[0], crop_size[1])
        ])

    def __call__(self, img, label, img2, label2):
        img, label = img.copy(), label.copy()
        transformed = self.cpm_transforms(image=img2, mask=label2)
        img2, label2 = transformed['image'], transformed['mask']
        
        n, cluster_mask = self.make_cluster_mask(label2)
        if n <= 1:
            return None, None
        
        cluster_mask2 = self.sample_patch(cluster_mask, n)
        
        rd_x = random.randint(0, self.size[0]-self.crop_size[0]-1)
        rd_y =random.randint(0, self.size[1]-self.crop_size[1]-1)
        
        img[rd_x:rd_x+self.crop_size[0], rd_y:rd_y+self.crop_size[1], :] = \
            img[rd_x:rd_x+self.crop_size[0], rd_y:rd_y+self.crop_size[1], :] * (1-cluster_mask2[:, :, np.newaxis]) + \
                cluster_mask2[:, :, np.newaxis] * img2
        label[rd_x:rd_x+self.crop_size[0], rd_y:rd_y+self.crop_size[1]] = \
            label[rd_x:rd_x+self.crop_size[0], rd_y:rd_y+self.crop_size[1]] * (1-cluster_mask2) + \
                cluster_mask2 * label2
                
        return img, label

    def make_cluster_mask(self, mask):
        ptr = 0
        merged_mask = np.zeros_like(mask)
        for c in self.include_classes:
            num_labels, labels = cv2.connectedComponents((mask==c).astype(np.uint8))
            if num_labels > 1:
                merged_mask += (labels!=0).astype('int32') * (labels+ptr)
                ptr += num_labels-1
        return ptr+1, merged_mask

    def sample_patch(self, cluster_mask, n):
        rd_index = np.random.randint(1, n, (min(n-1, self.npatch),))
        return np.isin(cluster_mask, rd_index).astype('int32')

    def patch_mix(self, img, label, src_img, src_label, src_cluster_mask):
        img, label = img.copy(), label.copy()
        src_cluster_mask = src_cluster_mask[:, :, np.newaxis]
        img = img * (1-src_cluster_mask) + src_img * src_cluster_mask
        label = label * (1-src_cluster_mask) + src_label * src_cluster_mask
        return img, label
    