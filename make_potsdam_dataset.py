from PIL import Image
import os
import numpy as np
from tqdm import tqdm

from utils.vis import rgb2label, potsdam_class_color_map


def build_dataset(
    src_img_path, src_gt_path, dest_img_path, dest_gt_path, 
    size=(1000, 1000)):
    # size: [H, W]
    names = sorted([n.split(".")[0].strip("_RGB") for n in os.listdir(src_img_path)])
    imgs = [os.path.join(src_img_path, n+"_RGB.tif") for n in names]
    labels = [os.path.join(src_gt_path, n+"_label.tif") for n in names]
    
    if not os.path.exists(dest_img_path): os.makedirs(dest_img_path)
    if not os.path.exists(dest_gt_path): os.makedirs(dest_gt_path)
    
    with tqdm(total=len(names)*(6000//size[0])*(6000//size[1])) as pbar:
        for name, imgp, labelp in zip(names, imgs, labels):
            img = np.array(Image.open(imgp))
            label = np.array(Image.open(labelp))
            
            # correct label mistakes of orginal data
            if name in ['top_potsdam_4_12', 'top_potsdam_6_7']:
                label[label<128] = 0
                label[label>128] = 255
            
            for i in range(6000//size[0]):
                for j in range(6000//size[1]):
                    slice_img = img[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1], :]
                    slice_gt = label[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1], :]
            
                    Image.fromarray(slice_img).save(os.path.join(dest_img_path, name+f"_{i+1}_{j+1}_RGB.tif"))
                    Image.fromarray(slice_gt).save(os.path.join(dest_gt_path, name+f"_{i+1}_{j+1}_label.tif"))
            
                    pbar.update(1)
    print("Images/Labels (split) finished")
                    

def resize_dataset(
        src_img_path, src_gt_path, dest_img_path, dest_gt_path, 
        size=(512, 512)
    ):
    
    if not os.path.exists(dest_img_path): os.makedirs(dest_img_path)
    if not os.path.exists(dest_gt_path): os.makedirs(dest_gt_path)
    
    img_names, gt_names = os.listdir(src_img_path), os.listdir(src_gt_path)
    with tqdm(total=len(img_names)) as pbar:
        for name in img_names:
            Image.open(
                os.path.join(src_img_path, name)).resize(
                    (size[1], size[0])).save(os.path.join(dest_img_path, name))
            pbar.update(1)
    print("Images (resize) finished")
    with tqdm(total=len(gt_names)) as pbar:
        for name in gt_names:
            label = np.array(Image.open(os.path.join(src_gt_path, name)))
            label = rgb2label(label, potsdam_class_color_map())
            Image.fromarray(label).resize(
                (size[1], size[0]), resample=Image.NEAREST).save(os.path.join(dest_gt_path, name))
            pbar.update(1)
    print("Labels (resize/rgb2label) finished")


if __name__ == "__main__":
    src_img_path = "./dataset/Potsdam/2_Ortho_RGB"
    src_gt_path = "./dataset/Potsdam/5_Labels_all"
    dest_img_path = "./dataset/Potsdam/images"
    dest_gt_path = "./dataset/Potsdam/labels"
    dest_img_path2 = "./dataset/Potsdam/images2"
    dest_gt_path2 = "./dataset/Potsdam/labels2"
    size = (1000, 1000) # (H, W)
    size2 = (512, 512)
    
    # build_dataset(src_img_path, src_gt_path, dest_img_path, dest_gt_path, size)
    resize_dataset(dest_img_path, dest_gt_path, dest_img_path2, dest_gt_path2, size2)