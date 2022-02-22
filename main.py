"""
Created on 20 Sep, 2020

@author: Xiaokun Zhang
"""

import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

import metric
from utils import collate_fn
from didn import DIDN
from dataset import load_data, RecSysDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='datasets/yoochoose1_64/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size 512')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden state size of gru module')
parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of item embedding')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for 100')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80,
                    help='the number of steps after which the learning rate decay')
parser.add_argument('--test', default=False, help='test')
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')

parser.add_argument('--position_embed_dim', type=int, default=64, help='the dimension of position embedding')
parser.add_argument('--max_len', type=float, default=19, help='max length of input session')
parser.add_argument('--alpha1', type=float, default=0.1, help='degree of alpha pooling')
parser.add_argument('--alpha2', type=float, default=0.1, help='degree of alpha pooling')
parser.add_argument('--alpha3', type=float, default=0.1, help='degree of alpha pooling')
parser.add_argument('--pos_num', type=int, default=2000, help='the number of position encoding')
parser.add_argument('--neighbor_num', type=int, default=5, help='the number of neighboring sessions')

args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.cuda.set_device(1)


def main():
    print('Loading data...')
    train, valid, test = load_data(args.dataset_path, valid_portion=args.valid_portion)

    train_data = RecSysDataset(train)
    valid_data = RecSysDataset(valid)
    test_data = RecSysDataset(test)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.dataset_path.split('/')[-2] == 'diginetica':
        n_items = 43098
    elif args.dataset_path.split('/')[-2] in ['yoochoose1_64', 'yoochoose1_4']:
        n_items = 37484
    else:
        raise Exception('Unknown Dataset!')
    model = DIDN(n_items, args.hidden_size, args.embed_dim, args.batch_size, args.max_len, args.position_embed_dim, args.alpha1, args.alpha2, args.alpha3,args.pos_num, args.neighbor_num).to(device)

    if args.test:
        ckpt = torch.load('latest_checkpoint.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        recall, mrr = validate(test_loader, model)
        # print("Test: Recall@{}: {:.4f}, MRR@{}: {:.4f}".format(args.topk, recall, args.topk, mrr))
        print("Test: Recall@{}\t MRR@{}\t".format(args.topk, args.topk))
        print("Test: {:.4f}\t {:.4f}".format(recall * 100, mrr * 100))
        return

    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        scheduler.step(epoch=epoch)
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr=200)

        recall, mrr = validate(valid_loader, model)
        print('Epoch {} validation: Recall@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, args.topk, recall, args.topk,
                                                                                 mrr))

        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')


def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq = seq.to(device)
        target = target.to(device)
        # neighbors = neighbors.to(device)

        optimizer.zero_grad()
        outputs = model(seq, lens)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                  % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                     len(seq) / (time.time() - start)))

        start = time.time()


def validate(valid_loader, model):
    model.eval()
    recalls = []
    mrrs = []
    with torch.no_grad():
        for seq, target, lens in tqdm(valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            # neighbors = neighbors.to(device)
            # case study
            # case_seq = seq.permute(1,0)
            # np.save('seq.npy', case_seq.cpu())
            # np.save('label.npy', target.cpu())
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim=1)
            recall, mrr = metric.evaluate(logits, target, k=args.topk)
            recalls.append(recall)
            mrrs.append(mrr)

    mean_recall = np.mean(recalls)
    mean_mrr = np.mean(mrrs)
    return mean_recall, mean_mrr


class Set_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        cross_entropy = F.nll_loss(input, target)

        return cross_entropy


if __name__ == '__main__':
    main()
