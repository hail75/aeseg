import os.path as osp
import time

import kagglehub
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from tqdm import tqdm

from architecture import UNet
from loss import SegmentationLoss
from dataset import PotsdamDataset, val_aug
from evaluator import Evaluator


def save_model(model, optimizer, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def train(
    model,
    loss_function,
    optimizer,
    device,
    train_dataloader, 
    valid_dataloader, 
    learing_rate_scheduler, 
    epoch
):
    print(f"Start epoch #{epoch+1}, learning rate for this epoch: {learing_rate_scheduler.get_last_lr()}")
    start_time = time.time()
    train_loss_epoch = 0
    test_loss_epoch = 0
    model.to(device)
    model.train()
    metrics_train = Evaluator(num_class=6)
    metrics_val = Evaluator(num_class=6)

    for i,input in enumerate(tqdm(train_dataloader)):
        # Load data into GPU
        data=input['img']
        masks_true = input['gt_semantic_seg']
        data,mask = data.to(device),masks_true.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        # Backpropagation, compute gradients
        loss = loss_function(prediction, mask.long())
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
        loss.backward()

        # Apply gradients
        optimizer.step()

        # Save loss
        train_loss_epoch += loss.item()
    print(f"Done epoch #{epoch+1}, time for this epoch: {time.time()-start_time}s")
    train_loss_epoch /= (i + 1)
    mIoU = np.nanmean(metrics_train.Intersection_over_Union()[:-1])
    F1 = np.nanmean(metrics_train.F1()[:-1])
    OA = np.nanmean(metrics_train.OA())
    iou_per_class = metrics_train.Intersection_over_Union()
    train_eval_value =  (iou_per_class,mIoU,F1,OA)
    metrics_train.reset()
    # Evaluate the validation set
    model.eval()
    with torch.no_grad():
        for input in tqdm(valid_dataloader):
            data=input['img'].to(device)
            mask = input['gt_semantic_seg'].to(device)
            prediction = model(data)
            test_loss = loss_function(prediction, mask.long())
            pre_mask = nn.Softmax(dim=1)(prediction)
            pre_mask = pre_mask.argmax(dim=1)
            test_loss_epoch += test_loss.item()
            for i in range(mask.shape[0]):
                metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
    test_loss_epoch /= (i + 1)
    mIoU = np.nanmean(metrics_val.Intersection_over_Union()[:-1])
    F1 = np.nanmean(metrics_val.F1()[:-1])
    OA = np.nanmean(metrics_val.OA())
    iou_per_class_val = metrics_val.Intersection_over_Union()
    eval_value =  (iou_per_class_val,mIoU,F1,OA)
    metrics_val.reset()
    return train_loss_epoch, train_eval_value, test_loss_epoch, eval_value

def main():
    num_classes = 6

    # Number of epoch
    epochs = 30

    # Hyperparameters for training 
    learning_rate = 8e-04
    batch_size = 8

    # Model path
    checkpoint_path = 'checkpoints/'

    # Use the model as your desire
    model = UNet(n_class=num_classes)
    loss_function = SegmentationLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.6)
    dataset_path = kagglehub.dataset_download("trito12/potsdam-dataset")
    dataset = PotsdamDataset(
        data_root=dataset_path,
        mode='train',
        mosaic_ratio=0,
        transform=val_aug
    )
    train_size = 0.35
    valid_size = 0.65
    dumb1 = 0.15
    dumb2 = 0.85
    train_length = round(train_size * len(dataset))
    valid_length = round(valid_size * len(dataset))
    train_set, dumb0 = random_split(dataset, [train_length, valid_length])
    val_set, xxx = random_split(dumb0, [dumb1, dumb2])
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    result = train(
        model,
        loss_function,
        optimizer,
        device,
        train_dataloader, 
        valid_dataloader, 
        learning_rate_scheduler, 
        epochs
    )
    print('----------------------')
    print('Training finished')
    print('----------------------')
    print('Metrics:')
    print('Train loss:', result[0])
    print('Train mIoU:', result[1][1])
    print('Train F1:', result[1][2])
    print('Train OA:', result[1][3])
    print('Val loss:', result[2])
    print('Val mIoU:', result[3][1])
    print('Val F1:', result[3][2])
    print('Val OA:', result[3][3])
    print('----------------------')
    print('Saving model...')
    save_model(model, optimizer, osp.join(checkpoint_path, 'unet.pth'))
    print('Model saved')
    print('----------------------')