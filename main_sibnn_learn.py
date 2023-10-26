import argparse
import os
import shutil
import time
import math
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import model_list as models
import util

# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)

import sys
import gc

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18_sibnn_bireal_learn',
                    help='model architecture (default: alexnet)')
parser.add_argument('--data', metavar='DATA_PATH', default='./data/',
                    help='path to imagenet data (default: ./data/)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_policy', default='cosine',
                    help='lr policy')
parser.add_argument('--momentum', default=0.90, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--power', default=1.0, type=float,
                    metavar='P', help='power for poly learning-rate decay')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', 
                    help='path to pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0

# define global bin_op
bin_op = None

def main():
    global args, best_prec1
    args = parser.parse_args()

    args.results_dir = './checkpoint'
    args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            c = float(m.weight.data[0].nelement())
            m.weight.data = m.weight.data.normal_(0, 1.0/c)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data = m.weight.data.zero_().add(1.0)

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.pretrained))
        state_dict = torch.load(args.pretrained)

        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
            new_state_dict = OrderedDict()
            for key, val in state_dict.items():
                if key.startswith('module.'):
                    key = key[7:]
                new_state_dict[key] = val
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    learnable_thresh = []
    other_params = []
    for key, value in model.named_parameters():
        if 'grad_scale' in key or 'grad_bias' in key:
            learnable_thresh.append(value)
        else:
            other_params.append(value)

    params = [
        {"params": learnable_thresh, 'weight_decay': 0.0, 'lr': args.lr},
        {"params": other_params, 'weight_decay': args.weight_decay, "lr": args.lr},
    ]
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                             weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    input_size = 224
    if 'alexnet' in args.arch:
        input_size = 227

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    args.epoch_size = len(train_dataset) // args.batch_size

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print(args)
    print(model)

    # define the binarization operator
    global bin_op
    bin_op = util.BinOp(model)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch)
        #adjust_learning_rate_epoch_poly(optimizer, epoch, args.epochs)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, path=save_path)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    if args.lr_policy == 'step':
        local_lr = adjust_learning_rate(optimizer, epoch, args)
    elif args.lr_policy == 'epoch_poly':
        local_lr = adjust_learning_rate_epoch_poly(optimizer, epoch, args)

    for i, (input, target) in enumerate(train_loader):
        global_iter = epoch * args.epoch_size + i
        if args.lr_policy == 'iter_poly':
            local_lr = adjust_learning_rate_poly(optimizer, global_iter, args)
        elif args.lr_policy == 'cosine':
            local_lr = adjust_learning_rate_cosine(optimizer, global_iter, args)
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # process the weights including binarization
        bin_op.binarization()
        
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight_new()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR: {lr: .6f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=local_lr))
        gc.collect()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bin_op.binarization()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
    bin_op.restore()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, path='./', filename='checkpoint.pth.tar'):

    saved_path = os.path.join(path, filename)
    best_path = os.path.join(path, 'model_best.pth.tar')
    torch.save(state, saved_path)
    if is_best:
        shutil.copyfile(saved_path, best_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 25 epochs"""
    if epoch<80:
        lr = args.lr * (0.1 ** (epoch // 40))
    else:
        lr = 0.01 * args.lr * (0.1 ** ((epoch-80) // 20))
    #print 'Learning rate:', lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_epoch_poly(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 25 epochs"""
    lr = args.lr * ((1 - epoch * 1.0 / args.epochs) ** args.power)
    #print 'Learning rate:', lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_poly(optimizer, global_iter, args):
    """Sets the learning rate to the initial LR decayed by 10 every 25 epochs"""
    lr = args.lr * ((1 - global_iter * 1.0 / (args.epochs * args.epoch_size)) ** args.power)
    #print 'Learning rate:', lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_cosine(optimizer, global_iter, args):
    # warmup_lr = args.lr * args.warmup_lr_multiplier
    max_iter = args.epochs * args.epoch_size
    # warmup_iter = args.warmup_epochs * args.epoch_size
    # if global_iter < warmup_iter:
    #     slope = (args.lr - warmup_lr) / warmup_iter
    #     lr = slope * global_iter + warmup_lr
    # else:
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * (global_iter) / (max_iter)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
