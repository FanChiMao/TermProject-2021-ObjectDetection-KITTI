import time
import logging
import os, sys, math
import argparse
from collections import deque
import datetime

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
from tool.tv_reference.utils import collate_fn as test_collate
from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4
from tool.darknet2pytorch import Darknet

from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator


@torch.no_grad()
def evaluate1(model, data_loader, cfg, device, logger=None, **kwargs):
    """ finished, tested
    """
    # cpu_device = torch.device("cpu")
    model.eval()
    # header = 'Test:'

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"], bbox_fmt='coco')
    imgIds = sorted(coco.getImgIds())
    for images, targets in data_loader:
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = model(model_input)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # outputs = outputs.cpu().detach().numpy()
        res = {}
        # for img, target, output in zip(images, targets, outputs):
        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes = boxes.squeeze(2).cpu().detach().numpy()
            boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            boxes[..., 0] = boxes[..., 0] * img_width
            boxes[..., 1] = boxes[..., 1] * img_height
            boxes[..., 2] = boxes[..., 2] * img_width
            boxes[..., 3] = boxes[..., 3] * img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # confs = output[...,4:].copy()
            confs = confs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = np.max(confs, axis=1).flatten()
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        # res是模型預測出來的
        coco_evaluator.update(res)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()



    return coco_evaluator,


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
    #                     help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='0',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str,
                                                                                 # test/new_images/
                        default='C:/Users/Lab722 BX/Desktop/term project/dataset v4/test/new_images/',
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default='./yolov4.conv.137.pth', help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=8, help='dataset classes')
    parser.add_argument('-train_label_path', dest='train_label', type=str, default='train.txt', help="train label path")
    parser.add_argument(
        '-optimizer', type=str, default='adam',
        help='training optimizer',
        dest='TRAIN_OPTIMIZER')
    parser.add_argument(
        '-iou-type', type=str, default='iou',
        help='iou type (iou, giou, diou, ciou)',
        dest='iou_type')
    parser.add_argument(
        '-keep-checkpoint-max', type=int, default=10,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)

    return edict(cfg)


if __name__ == "__main__":
    cfg = get_args(**Cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kitti_weight = 'checkpoints/kitti_best_v4.pth'

    test_dataset = Yolo_dataset('D:/PycharmProjects/PyTorch-YOLOv4-kitti-master/test.txt', cfg, train=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=8,
                             pin_memory=True, drop_last=True, collate_fn=test_collate)

    model = Yolov4(yolov4conv137weight=kitti_weight, n_classes=8, inference=True)
    pretrained_dict = torch.load(kitti_weight, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict)

    model.to(device)
    evaluator = evaluate1(model=model, data_loader=test_loader, cfg=cfg, device=device)

