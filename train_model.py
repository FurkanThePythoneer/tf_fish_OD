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
from dataset import FishDataset, get_train_transform, get_valid_transform
from madgrad import MADGRAD




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

seed_everything(42)

def get_model():
    model = MyModel(
    	backbone_name="resnet101d",
    	imagenet_pretrained=False,
    	num_classes=2,
    	in_features=2048,
    	backbone_pretrained_path=None, 
    	backbone_pretrained_cls_num_classes=None,
    	model_pretrained_path=None,
    	model_pretrained_cls_num_classes=2)
    return model.to(device)

dl_train, dl_val = prepare_dataset()
model = get_model()



params = [p for p in model.parameters() if p.requires_grad]
optimizer = MADGRAD(params, lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS-1)


n_batches, n_batches_val = len(dl_train), len(dl_val)
validation_losses = []


for epoch in range(NUM_EPOCHS):
    time_start = time.time()
    loss_accum = 0
    
    for batch_idx, (images, targets) in enumerate(dl_train, 1):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Predict
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_accum += loss_value

        # Back-prop
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Validation 
    val_loss_accum = 0
        
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dl_val, 1):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            val_loss_dict = model(images, targets)
            val_batch_loss = sum(loss for loss in val_loss_dict.values())
            val_loss_accum += val_batch_loss.item()
    
    # Logging
    val_loss = val_loss_accum / n_batches_val
    train_loss = loss_accum / n_batches
    validation_losses.append(val_loss)
    
    # Save model
    chk_name = f'fasterrcnn_resnet101d-e{epoch}.bin'
    torch.save(model.state_dict(), chk_name)
    
    
    elapsed = time.time() - time_start
    
    print(f"[Epoch {epoch+1:2d} / {NUM_EPOCHS:2d}] Train loss: {train_loss:.3f}. Val loss: {val_loss:.3f} --> {chk_name}  [{elapsed:.0f} secs]")   





