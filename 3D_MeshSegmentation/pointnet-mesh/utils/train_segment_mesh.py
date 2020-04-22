from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset_mesh import PhenixDataset
from pointnet.model import PointNetMesh, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pdb
import torch.nn as nn
import shutil
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument(
    '--batchSize', type=int, default=1, help='input batch size')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
#'/home/nagaharish/Downloads/pointnet.pytorch-master/data_mesh/'
dataset = PhenixDataset(dataset_path=opt.dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
testdata = PhenixDataset(dataset_path=opt.dataset, split='test')
testdataloader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False, num_workers=1)
blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetMesh(k=1, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize
criterion = nn.BCELoss()
# pdb.set_trace()
if os.path.isdir(opt.outf):
    shutil.rmtree(opt.outf)
    os.mkdir(opt.outf)
else:
    os.mkdir(opt.outf)
loss_values = []
for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        npts = points.shape[1]
        # pdb.set_trace()
        # print(points.shape)
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        pred = pred.view(-1)
        target = target.view(-1, 1)[:, 0]
        # pdb.set_trace()
        target = target.type(torch.cuda.FloatTensor)
        loss = criterion(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        # pred_choice = pred.data.max(1)[1]
        pred_choice = pred.data>=0.5
        pred_choice = pred_choice.type(torch.cuda.FloatTensor)
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(npts)))
    loss_values.append(loss / len(dataset))
    if (epoch+1)%50==0:
        torch.save(classifier.state_dict(), '%s/mesh_model_%d.pth' % (opt.outf, epoch))

plt.plot(loss_values)
plt.savefig('training_loss.png')

# ## benchmark mIOU
# shape_ious = []
dir_lst = os.listdir(os.path.join(opt.dataset,'test'))
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    # pdb.set_trace()
    pred = pred.view(-1)
    target = target.view(-1, 1)[:, 0]
    pred_choice = pred.data>=0.5
    pred_choice = pred_choice.type(torch.cuda.FloatTensor)
    pred_numpy = pred_choice.cpu().numpy()
    labl_fl = open(dir_lst[i]+'.txt','w')
    # pdb.set_trace()
    for val in pred_numpy:
        labl_fl.write(str(int(val)))
        labl_fl.write('\n')
    labl_fl.close()
#     pred_choice = pred.data.max(2)[1]

#     pred_np = pred_choice.cpu().data.numpy()
#     target_np = target.cpu().data.numpy() - 1

#     for shape_idx in range(target_np.shape[0]):
#         parts = range(num_classes)#np.unique(target_np[shape_idx])
#         part_ious = []
#         for part in parts:
#             I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#             U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
#             if U == 0:
#                 iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
#             else:
#                 iou = I / float(U)
#             part_ious.append(iou)
#         shape_ious.append(np.mean(part_ious))

# print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))