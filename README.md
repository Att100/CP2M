# CP2M: Clustered-Patch-Mixed Mosaic Augmentation for Aerial Image Segmentation


## 1. Summery

Remote sensing image segmentation is pivotal for earth observation, underpinning applications such as environmental monitoring and urban planning. Due to the limited annotation data available in remote sensing images, numerous studies have focused on data augmentation as a means to alleviate overfitting in deep learning networks. However, some existing data augmentation strategies rely on simple transformations that may not sufficiently enhance data diversity or model generalization capabilities. This paper proposes a novel augmentation strategy, Clustered-Patch-Mixed Mosaic (CP2M), designed to address these limitations. CP2M integrates a Mosaic augmentation phase with a clustered patch mix phase. The former stage constructs a new sample from four random samples, while the latter phase uses the connected component labeling algorithm to ensure the augmented data maintains spatial coherence and avoids introducing irrelevant semantics when pasting random patches. Our experiments on the ISPRS Potsdam dataset demonstrate that CP2M substantially mitigates overfitting, setting new benchmarks for segmentation accuracy and model robustness in remote sensing tasks.


**Overall Pipeline**
![](./assets/cp2m.svg)
**Segmentation Model**
![](./assets/cp2m-unet.svg)

## 2. Dependencies

### 2.1 PaddlePaddle

For CUDA 12

```
python -m pip install paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
```

### 2.2 Others

```
pip install numpy pillow pandas matplotlib seaborn wandb tqdm albumentations
```

### 2.3 Dataset

The offical website of ISPRS-Potsdam is [link](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx). You can download the data from [link](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx). Please place the data in `./dataset` following the structure:

```
.
├── Potsdam
│   ├── 2_Ortho_RGB
│   │   │── top_potsdam_4_12_RGB.tif
|   |   └── ...
│   └── 5_Labels_all
│       │── top_potsdam_4_12_label.tif
|       └── ...
└── README.md
```

Then please run `python make_potsdam_dataset.py` to generate the data for training and testing.

## 3. Usage

### 3.1 Train

```
usage: train.py [-h] [--img_path IMG_PATH] [--gt_path GT_PATH] [--percentage PERCENTAGE] [--epochs EPOCHS] [--batchsize BATCHSIZE] [--lr LR] [-aug]
                [--p_mosaic P_MOSAIC] [--p_cpm P_CPM] [--name NAME] [--key KEY] [--proj PROJ]

options:
  -h, --help            show this help message and exit
  --img_path IMG_PATH  
  --gt_path GT_PATH
  --percentage PERCENTAGE
  --epochs EPOCHS
  --batchsize BATCHSIZE
  --lr LR
  -aug
  --p_mosaic P_MOSAIC  # probability of applying MOSAIC augmentation
  --p_cpm P_CPM  # probability of applying CP2M augmentation
  --name NAME
  --key KEY
  --proj PROJ
```

**Train Using Default Setting**

```
python train.py --name <EXPPERIMENT NAME> --key <YOUR WANDB KEY> -aug
```

### 3.2 Test

```
python test.py --gt_path <PATH OF THE CHECKPOINT>
```


