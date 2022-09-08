import time
import shutil
import os
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from ccn_resnet_model import se_resnet101

class DefaultConfigs(object):
    # 1.string parameters
    train_dir = "H:/ImageNet/data/ImageNet2012/train100"
    val_dir = 'H:/ImageNet/data/ImageNet2012/val100'
    model_name = "ccn_resnet101_model/"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"

    # 2.numeric parameters
    epochs = 60
    start_epoch = 0
    batch_size = 64
    momentum = 0.9
    lr = 0.025
    weight_decay = 1e-4
    interval = 10
    workers = 4

    # 3.boolean parameters
    evaluate = False
    pretrained = False
    resume = False

device = "cuda" if torch.cuda.is_available() else "cpu"

best_acc = 0
config = DefaultConfigs()


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = config.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best):
    filename = config.weights + config.model_name + os.sep + "_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        message = config.best_models + config.model_name + os.sep + 'model_best.pth.tar'
        shutil.copyfile(filename, message)


def validate(val_loader, model, criterion,epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_id, (images, target) in enumerate(val_loader):
            images, target = images.to(device), target.to(device)
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (batch_id + 1) % config.interval == 0:
                progress.display(batch_id + 1)
        write.add_scalar('acc_top1',top1.avg.item(),global_step=epoch)
        write.add_scalar('acc_top5',top5.avg.item(),global_step=epoch)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()
    for batch_id, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images, target = images.to(device), target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_id + 1) % config.interval == 0:
            progress.display(batch_id + 1)
    write.add_scalar('train_acc_top1', top1.avg.item(), global_step=epoch)


def main():
    global best_acc

    # # 加载model，model是自己定义好的模型
    # resnet50 = models.resnet50(pretrained=True)

    model = se_resnet101()
    model.to(device)
    # # # 读取参数
    # pretrained_dict = resnet50.state_dict()
    #
    # model_dict = model.state_dict()
    # # 将pretrained_dict里不属于model_dict的键剔除掉
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 更新现有的model_dict
    # model_dict.update(pretrained_dict)
    # # 加载我们真正需要的state_dict
    # model.load_state_dict(model_dict)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    cudnn.benchmark = True

    if config.resume:
        checkpoint = torch.load(config.best_models +config.model_name+ "model_best.pth.tar")
        config.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(config.train_dir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=True,drop_last= True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(config.val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True,drop_last= True)

    if config.evaluate:
        validate(val_loader, model, criterion,config.epochs)
        return

    for epoch in range(config.start_epoch, config.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d]' % (epoch + 1, config.epochs))

        train(train_loader, model, criterion, optimizer, epoch)
        test_acc = validate(val_loader, model, criterion,epoch)

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            "model_name": config.model_name,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)


if __name__ == '__main__':
    write = SummaryWriter('./runs/ccn_resnet101_NEW')
    main()