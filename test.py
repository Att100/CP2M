import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
import argparse
from tqdm import tqdm

from models.unet import UNet
from utils.metrics import Metrics, per_class_precision_and_iou
from utils.data import Potsdam
from utils.vis import potsdam_class_map


def test(args):
    test_metrics = Metrics(6, (512, 512))
    test_dataset = Potsdam(
        img_path=args.img_path, 
        gt_path=args.gt_path, 
        dtsplit='test', p_mosaic=0, p_cpm=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=8)
    
    model = UNet(6, False)
    model.set_state_dict(paddle.load(args.ckpt))
    model.eval()
    
    with tqdm(total=len(test_loader)) as pbar:
        for i, (img, label) in enumerate(test_loader):
            pred = model(img)
            test_metrics.update(paddle.argmax(F.softmax(pred ,axis=1), axis=1), label)
            pbar.update(1)
            
    _test_metrics = test_metrics.total_values()
    print(f"Checkpoint: {args.ckpt} - miou: {_test_metrics['miou']}, acc: {_test_metrics['acc']}")

    per_class_precision, per_class_iou = per_class_precision_and_iou(model, test_loader, 6)
    _cname = potsdam_class_map()
    print(f"Checkpoint: {args.ckpt} - per-class-iou")
    for i in range(6):
        print("   - {}: {:.4f}".format(_cname[i], float(per_class_iou[i])))
    print(f"Checkpoint: {args.ckpt} - per-class-precision")
    for i in range(6):
        print("   - {}: {:.4f}".format(_cname[i], float(per_class_precision[i])))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="./dataset/Potsdam/images2")
    parser.add_argument("--gt_path", type=str, default="./dataset/Potsdam/labels2")
    parser.add_argument("--ckpt", type=str, default="./ckpts/unet_mbv2_b8_no_aug.pdparam")
    
    args = parser.parse_args()
    test(args)