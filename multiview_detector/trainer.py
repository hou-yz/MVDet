import time
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class MultiviewTrainer(BaseTrainer):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, 1)
            true_positive = (pred.eq(target) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = target.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            precision_s.update(precision)
            recall_s.update(recall)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, '
                      'precision: {:.1f}%, Recall: {:.1f}%, \tTime: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), precision_s.avg * 100, recall_s.avg * 100,
                    t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, '
              'precision: {:.1f}%, Recall: {:.1f}%, \tTime: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg

    def test(self, test_loader, device=0):
        self.model.eval()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(device), target.cuda(device)
            with torch.no_grad():
                output = self.model(data)
            pred = torch.argmax(output, 1)
            true_positive = (pred.eq(target) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = target.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            precision_s.update(precision)
            recall_s.update(recall)
            loss = self.criterion(output, target)
            losses += loss.item()

        print('Test, Loss: {:.6f}, precision: {:.1f}%, Recall: {:.1f}%'.format(losses / (len(test_loader) + 1),
                                                                                      precision_s.avg * 100,
                                                                                      recall_s.avg * 100))

        return losses / len(test_loader), precision_s.avg * 100
