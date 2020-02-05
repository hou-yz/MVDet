import time
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from .utils.meters import AverageMeter
from .utils.nms import nms


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres=0.1):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None, visualize=False):
        self.model.train()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        for batch_idx, (data, map_gt, imgs_gt) in enumerate(data_loader):
            data, map_gt = data.cuda(), map_gt.cuda()
            optimizer.zero_grad()
            map_res, imgs_res = self.model(data)

            loss = self.criterion(map_res, map_gt.to(map_res.device))
            for img_res, img_gt in zip(imgs_res, imgs_gt):
                loss += self.criterion(img_res, img_gt.to(img_res.device))
            loss.backward()
            optimizer.step()
            losses += loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)
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
                      'precision: {:.1f}%, Recall: {:.1f}%, \tTime: {:.3f}, maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), precision_s.avg * 100, recall_s.avg * 100,
                    t_epoch, map_res.max()))
                if visualize:
                    fig = plt.figure()
                    subplt0 = fig.add_subplot(211, title="output")
                    subplt1 = fig.add_subplot(212, title="target")
                    subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
                    subplt1.imshow(self.criterion._traget_transform(map_res, map_gt)
                                   .cpu().detach().numpy().squeeze())
                    plt.savefig('map.jpg')
                    plt.close(fig)

                    # visualizing the heatmap for per-view estimation
                    plt.imshow(imgs_res[0].detach().cpu().numpy().squeeze())
                    plt.savefig('cam0.jpg')
                pass

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, '
              'precision: {:.1f}%, Recall: {:.1f}%, \tTime: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100

    def test(self, test_loader):
        self.model.eval()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        for batch_idx, (data, map_gt, imgs_gt) in enumerate(test_loader):
            data, map_gt = data.cuda(), map_gt.cuda()
            with torch.no_grad():
                map_res, imgs_res = self.model(data)
            loss = self.criterion(map_res, map_gt.to(map_res.device))
            for img_res, img_gt in zip(imgs_res, imgs_gt):
                loss += self.criterion(img_res, img_gt.to(img_res.device))
            losses += loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

        print('Test, Loss: {:.6f}, precision: {:.1f}%, Recall: {:.1f}%'.format(losses / (len(test_loader) + 1),
                                                                               precision_s.avg * 100,
                                                                               recall_s.avg * 100))

        return losses / len(test_loader), precision_s.avg * 100


class BBOXTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target, _) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
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
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, log_interval=100, res_fpath=None):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        all_res_list = []
        t0 = time.time()
        for batch_idx, (data, target, (frame, pid, grid_x, grid_y)) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(data)
                output = F.softmax(output, dim=1)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()
            if res_fpath is not None:
                indices = output[:, 1] > self.cls_thres
                all_res_list.append(torch.stack([frame[indices].float(), grid_x[indices].float(),
                                                 grid_y[indices].float(), output[indices, 1].cpu()], dim=1))
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            len(test_loader), losses / (len(test_loader) + 1), 100. * correct / (correct + miss), t_epoch))

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.dirname(res_fpath) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, )
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy()
            np.savetxt(res_fpath, res_list, '%d')

        return losses / len(test_loader), correct / (correct + miss)

    def append_result(self, output_s, frame_s, x_s, y_s, frame_res_dict):
        for i in range(len(frame_s)):
            output, frame, x, y = output_s[i, 1].item(), frame_s[i].item(), x_s[i].item(), y_s[i].item()
            if output < self.cls_thres:
                continue
            cur_frame_dict = frame_res_dict[frame] if frame in frame_res_dict else {'scores': [], 'x_s': [], 'y_s': []}
            cur_frame_dict['scores'].append(output)
            cur_frame_dict['x_s'].append(x)
            cur_frame_dict['y_s'].append(y)
            frame_res_dict[frame] = cur_frame_dict
        pass
