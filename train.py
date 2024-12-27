import paddle
from tqdm import tqdm
import paddle.nn as nn
import argparse
import wandb
import os
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.optimizer import Adam
import matplotlib.pyplot as plt

from utils.data import Potsdam
from utils.metrics import Metrics
from utils.vis import get_vis_samples, potsdam_class_map
from models.unet import UNet


def get_wandb_images(args, model, sample_names=[], size=(512, 512)):
    img_paths = [os.path.join(args.img_path, n+"_RGB.tif") for n in sample_names]
    gt_paths = [os.path.join(args.gt_path, n+"_label.tif") for n in sample_names]
    imgs, preds, gts = get_vis_samples(model, img_paths, gt_paths, size, None)
    
    wandb_imgs = []
    for i, (img, pred, gt) in enumerate(zip(imgs, preds, gts)):
        wandb_imgs.append(
            wandb.Image(
                img,
                masks={
                    "prediction" : {"mask_data" : pred, "class_labels" : potsdam_class_map()},
                    "ground truth" : {"mask_data" : gt, "class_labels" : potsdam_class_map()}
                }
            )
        )
    return wandb_imgs


def run_fsl(args):
    wandb.login(key=args.key)
    wandb.init(
        project=args.proj,
        config={
            'epochs': args.epochs,
            'lr': args.lr,
            'percentage': args.percentage,
            'batchsize': args.batchsize,
            'dataset': 'Potsdam',
            'optimizer': 'Adam',
            'aug': args.aug,
            'p_mosaic': args.p_mosaic,
            'p_cpm': args.p_cpm,
            'size': (512, 512),
        }
    )
    wandb.define_metric("epoch_metrics/step")
    wandb.define_metric("epoch_metrics/*", step_metric="epoch_metrics/step")
    
    train_set = Potsdam(
        args.img_path, args.gt_path, 'train', 
        p_mosaic=args.p_mosaic if args.aug else 0.0,
        p_cpm=args.p_cpm if args.aug else 0.0)
    val_set = Potsdam(args.img_path, args.gt_path, 'val', p_mosaic=0, p_cpm=0)
    train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=8)
    train_metrics, val_metrics = Metrics(6, (512, 512)), Metrics(6, (512, 512))
    
    model = UNet(6, True)
    optimizer = Adam(parameters=model.parameters(), learning_rate=args.lr, weight_decay=4e-5)
    ce = nn.CrossEntropyLoss(axis=1)
    
    n_train_steps, n_val_steps = len(train_loader), len(val_loader)
    
    for e in range(args.epochs):
        model.train()
        with tqdm(total=n_train_steps, desc=f'Epoch {e+1}/{args.epochs}', unit='batch') as pbar:
            for i, (img, label) in enumerate(train_loader):
                optimizer.clear_grad()
                pred = model(img)
                loss = ce(pred, label)
                loss.backward()
                optimizer.step()
                
                train_metrics.update(paddle.argmax(F.softmax(pred ,axis=1), axis=1), label)
                miou, acc = train_metrics.current_miou, train_metrics.current_acc
                wandb.log({'batch_metrics/train_miou': miou, 'batch_metrics/train_acc': acc})
                pbar.set_description(f'Epoch {e+1}/{args.epochs}, Iter {i+1}/{n_train_steps} - loss: {float(loss):.4f}, mIoU: {miou:.4f}, accuracy: {acc:.4f}')
                pbar.update(1)
            
        model.eval()
        with tqdm(total=n_val_steps, desc=f'Epoch {e+1}/{args.epochs} (Val)', unit='batch') as pbar:
            for i, (img, label) in enumerate(val_loader):
                pred = model(img)
                
                val_metrics.update(paddle.argmax(F.softmax(pred ,axis=1), axis=1), label)
                miou, acc = val_metrics.current_miou, val_metrics.current_acc
                pbar.set_description(f'Epoch {e+1}/{args.epochs} (Val), Iter {i+1}/{n_val_steps} - mIoU: {miou:.4f}, accuracy: {acc:.4f}')
                pbar.update(1)
        
        _train_metrics, _val_metrics = train_metrics.total_values(), val_metrics.total_values()
        train_miou, train_acc = _train_metrics['miou'], _train_metrics['acc']
        val_miou, val_acc = _val_metrics['miou'], _val_metrics['acc']
        
        wandb.log({
            'epoch_metrics/step': e+1, 
            'epoch_metrics/val_miou': val_miou, 'epoch_metrics/val_acc': val_acc, 
            'epoch_metrics/train_miou': train_miou, 'epoch_metrics/train_acc': train_acc,
            'epoch_metrics/predictions': get_wandb_images(
                args, model,
                [
                    'top_potsdam_2_13_1_1', 'top_potsdam_2_14_1_1', 
                    'top_potsdam_3_13_1_1', 'top_potsdam_3_14_1_1'])
        })
        print(f"-- Epoch {e+1}/{args.epochs} Train Metrics:: mIoU: {train_miou:.4f}, accuracy: {train_acc:.4f} --")
        print(f"-- Epoch {e+1}/{args.epochs} Val Metrics:: mIoU: {val_miou:.4f}, accuracy: {val_acc:.4f} --")
        
        train_metrics.reset()
        val_metrics.reset()

    paddle.save(model.state_dict(), f"./ckpts/{args.name}.pdparam")
            
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # training configs
    parser.add_argument("--img_path", type=str, default="./dataset/Potsdam/images2")
    parser.add_argument("--gt_path", type=str, default="./dataset/Potsdam/labels2")
    parser.add_argument("--percentage", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("-aug", action='store_true', default=False)
    parser.add_argument("--p_mosaic", type=float, default=0.7)
    parser.add_argument("--p_cpm", type=float, default=0.8)
    parser.add_argument("--name", type=str, default="baseline")
    
    # wandb configs
    parser.add_argument("--key", type=str, default="-1")
    parser.add_argument("--proj", type=str, default="ICLR2024_Workshop")
    
    args = parser.parse_args()
    run_fsl(args)


