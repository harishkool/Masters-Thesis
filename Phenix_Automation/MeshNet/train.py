import copy
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from config import get_train_config
from data import ModelNet40
from data.Phenix import PhenixDataset
from models import MeshNet
from utils import append_feature, calculate_map
import pdb

cfg = get_train_config()
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']


data_set = {
    x: PhenixDataset(dataset_path = 'data/phenix_meshnet', split=x) for x in ['train', 'test']
}
data_loader = {
    x: data.DataLoader(data_set[x], batch_size=1, num_workers=1, shuffle=False, pin_memory=False)
    for x in ['train', 'test']
}


def train_model(model, criterion, optimizer, scheduler, cfg):

    best_acc = 0.0
    best_map = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg['max_epoch']):

        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
        print('-' * 60)

        for phrase in ['train', 'test']:

            if phrase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            ft_all, lbl_all = None, None
            if phrase=='train':
                for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader[phrase]):

                    running_loss = 0.0
                    running_corrects = 0

                    optimizer.zero_grad()

                    centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
                    corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
                    normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
                    neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))
                    targets = Variable(torch.cuda.LongTensor(targets.cuda()))

                    with torch.set_grad_enabled(phrase == 'train'):
                        outputs = model(centers, corners, normals, neighbor_index)
                        # pdb.set_trace()
                        targets = targets.view(-1)
                        outputs = outputs.view(-1,2)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, targets)

                        if phrase == 'train':
                            loss.backward()
                            optimizer.step()

                        # if phrase == 'test':
                        #     ft_all = append_feature(ft_all, feas.detach())
                        #     lbl_all = append_feature(lbl_all, targets.detach(), flaten=True)
                        # pdb.set_trace()
                        running_loss += loss.item() * centers.size(0)
                        running_corrects += torch.sum(preds == targets.data)
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phrase, loss, float(running_corrects)/len(targets)))
                    
            # epoch_loss = running_loss / len(data_set[phrase])
            # epoch_loss = running_loss
            # epoch_acc = running_corrects.double() / len(data_set[phrase])

            # if phrase == 'train':
                # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phrase, epoch_loss, epoch_acc))

            if phrase == 'test':
                with torch.no_grad():
                    for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader[phrase]):
                        centers = torch.cuda.FloatTensor(centers.cuda())
                        corners = torch.cuda.FloatTensor(corners.cuda())
                        normals = torch.cuda.FloatTensor(normals.cuda())
                        neighbor_index = torch.cuda.LongTensor(neighbor_index.cuda())
                        targets = torch.cuda.LongTensor(targets.cuda())

                        outputs = model(centers, corners, normals, neighbor_index)
                        # pdb.set_trace()
                        targets = targets.view(-1)
                        outputs = outputs.view(-1,2)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, targets)
                        corrects = torch.sum(preds == targets.data)
                        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phrase, loss, float(corrects)/len(targets)))
                # model = model.train()
    dir_lst = os.listdir('data/phenix_meshnet/test')
    with torch.no_grad():
        for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader['test']):
                centers = torch.cuda.FloatTensor(centers.cuda())
                corners = torch.cuda.FloatTensor(corners.cuda())
                normals = torch.cuda.FloatTensor(normals.cuda())
                neighbor_index = torch.cuda.LongTensor(neighbor_index.cuda())
                targets = torch.cuda.LongTensor(targets.cuda())

                outputs = model(centers, corners, normals, neighbor_index)
                # pdb.set_trace()
                targets = targets.view(-1)
                outputs = outputs.view(-1,2)
                _, preds = torch.max(outputs, 1)
                fl = open(dir_lst[i]+'.txt','w')
                preds = preds.cpu().numpy()
                for pred in preds:
                    fl.write(str(pred))
                    fl.write('\n')
                fl.close()
    return model


if __name__ == '__main__':

    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    model.cuda()
    # model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])

    best_model_wts = train_model(model, criterion, optimizer, scheduler, cfg)
    # torch.save(best_model_wts, os.path.join(cfg['ckpt'], 'MeshNet_best.pkl'))
