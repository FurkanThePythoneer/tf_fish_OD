import cv2
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import random
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from models import MyModel
from timm.utils.model_ema import ModelEmaV2

from dataset import FishDataset, get_train_transform, get_valid_transform
from madgrad import MADGRAD

from mean_average_precision import MetricBuilder




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_EPOCHS = 20

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('Done seeding...')
seed_everything(42)


def collate_fn(batch):
    return tuple(zip(*batch))

def prepare_dataset():
    df = pd.read_csv("/kaggle/input/reef-cv-strategy-subsequences-dataframes/train-validation-split/train-0.1.csv")

    # Turn annotations from strings into lists of dictionaries
    df['annotations'] = df['annotations'].apply(eval)

    # Create the image path for the row
    df['image_path'] = "video_" + df['video_id'].astype(str) + "/" + df['video_frame'].astype(str) + ".jpg"

    df_train, df_val = df[df['is_train']], df[~df['is_train']]

    df_train = df_train[df_train.annotations.str.len() > 0 ].reset_index(drop=True)
    df_val = df_val[df_val.annotations.str.len() > 0 ].reset_index(drop=True)

    ds_train = FishDataset(df_train, get_train_transform(image_sizes=[512,512]))
    ds_valid = FishDataset(df_val, get_valid_transform(image_sizes=[512,512]))



    dl_train = DataLoader(ds_train, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)
    dl_val = DataLoader(ds_valid, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)


    return dl_train, dl_val
# ---


dl_train, dl_val = prepare_dataset()

model = MyModel(
    backbone_name="resnet101d",
    imagenet_pretrained=False,
    num_classes=2,
    in_features=2048,
    backbone_pretrained_path=None, 
    backbone_pretrained_cls_num_classes=None,
    model_pretrained_path=None,
    model_pretrained_cls_num_classes=2)

model_ema = ModelEmaV2(model, decay=0.997, device=device)
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS-1)

scaler = torch.cuda.amp.GradScaler()

n_batches, n_batches_val = len(dl_train), len(dl_val)
validation_losses = []


ema_val_map_max = 0

DEVICE = device

for epoch in range(NUM_EPOCHS):
    step = 0
    
    model.train()
    scheduler.step()

    time_start = time.time()
    loss_accum = 0
    train_loss = []
    
    for batch_idx, (images, targets) in enumerate(dl_train, 1):
        
        images = list(image.float().to(device) for image in images)
        targets = [{k: v.to(torch.float32).to(device) if "box" in k else v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()


        with torch.cuda.amp.autocast():
            det_loss_dict = model(images, targets)
            loss = sum(l for l in det_loss_dict.values())
            train_loss.append(loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        model_ema.update(model)

        step += 1
        if step % 100 == 0:
            print('Step: [{}] | Loss: [{}] | LR: [{}]'.format(step, loss, optimizer.param_groups[0]['lr']))
    
    train_loss = np.mean(train_loss)
    model.eval()
    model_ema.eval()


    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
    eval_metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    for batch_idx, (images, targets) in enumerate(dl_val, 1):
        images = list(image.float().to(device) for image in images)
        targets = [{k: v.to(torch.float32).to(device) if "box" in k else v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(), torch.no_grad():
            det_outputs = model(images, targets)
            ema_det_outputs = model_ema.module(images, targets)

            for t, d, ed in zip(targets, det_outputs, ema_det_outputs):
                gt_boxes = t['boxes'].data.cpu().numpy()
                gt_boxes = np.hstack((gt_boxes, np.zeros((gt_boxes.shape[0], 3), dtype=gt_boxes.dtype)))

                det_boxes = d['boxes'].data.cpu().numpy()
                det_scores = d['scores'].data.cpu().numpy()
                det_scores = det_scores.reshape(det_scores.shape[0], 1)
                det_pred = np.hstack((det_boxes, np.zeros((det_boxes.shape[0], 1), dtype=det_boxes.dtype), det_scores))
                metric_fn.add(det_pred, gt_boxes)

                ema_det_boxes = ed['boxes'].data.cpu().numpy()
                ema_det_scores = ed['scores'].data.cpu().numpy()
                ema_det_scores = ema_det_scores.reshape(ema_det_scores.shape[0], 1)
                ema_det_pred = np.hstack((ema_det_boxes, np.zeros((ema_det_boxes.shape[0], 1), dtype=ema_det_boxes.dtype), ema_det_scores))
                eval_metric_fn.add(ema_det_pred, gt_boxes)

    val_map = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1), mpolicy='soft')['mAP']
    ema_val_map = eval_metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1), mpolicy='soft')['mAP']

    print('train loss: {:.5f} | val_map: {:.5f} | ema_val_map: {:.5f}'.format(train_loss, val_map, ema_val_map))
    
    if ema_val_map > ema_val_map_max:
        print("Ema val map improved from [{}] to [{}] - Saving model.".format(ema_val_map_max, ema_val_map))
        ema_val_map_max = ema_val_map
        torch.save(model_ema.module.state_dict(), "fasterrccn.pth")

