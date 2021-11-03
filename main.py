'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from logging import getLogger
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
import git
#mlfow settings
from mlflow import log_metric, log_param, log_artifacts
import mlflow
mlflow.set_tracking_uri("http://192.168.11.21:5000")
mlflow.set_experiment("cifar10-experiment")
#logging settings
logging.basicConfig(filename="log.txt", level=logging.INFO)
logger = getLogger(__name__)
# DDP settings
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()
# hparam settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--network', default="ResNet18", type=str, help='network type [ResNet18|PreActResNet18|'\
                    'GoogLeNet|DenseNet121|ResNeXt29_2x64d|MobileNet|MobileNetV2'\
                    'DPN92|huffleNetG2|SENet18|EfficientNetB0|RegNetX_200MF|SimpleDLA')
parser.add_argument('--world_size', default=2, type=int, help='gpu num')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=20, type=int, help='epoch')
parser.add_argument('--batchsize', default=256, type=int, help='epoch')
parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
parser.add_argument('--debug',action='store_true',help='debugging mode')
args = parser.parse_args()
if not args.debug:
    repo = git.Repo("./")
    uncommitted = repo.is_dirty()
    if uncommitted:
        print("コミットしてから実行してください")
        exit()
    
#main function
def train_and_test(rank, world_size):
    if rank ==0:
        for k,v in vars(args).items():
            log_param(k,v)
    def log(message):
        if rank == 0:
            print(message)
            logger.info(message)

    setup(rank, world_size)
    device = rank
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    best_epoch = 0

    # Data
    log('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_sampler = DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=False, num_workers=2, sampler=train_sampler)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_sampler = DistributedSampler(testset)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batchsize, shuffle=False, num_workers=2, sampler=test_sampler)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    log('==> Building model..')
    net = globals()[args.network]()
    net = net.to(device)
    net = DDP(net, device_ids=[rank])

    if args.resume:
        # Load checkpoint.
        log('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    def train(epoch):
        log('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:
                global_steps = batch_idx + epoch*len(trainloader)
                if rank == 0:
                    log_metric("train_Loss",(train_loss/(batch_idx+1)), global_steps)
                    log_metric("train_Acc",correct/total, global_steps)
                log('global_step %d|epoch %d|step %d|Loss: %.3f | Acc: %.3f (%d/%d)'
                        % (global_steps,epoch,batch_idx,train_loss/(batch_idx+1), correct/total, correct, total))
    def test(epoch):
        nonlocal best_acc
        nonlocal best_epoch
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                if batch_idx % 10 == 0:
                    if rank == 0:
                        global_steps = batch_idx + epoch*len(trainloader)
                        log_metric("test_loss",(test_loss/(batch_idx+1)), global_steps)
                        log_metric("test_Acc",correct/total, global_steps)
                    log('Loss: %.3f | Acc: %.3f (%d/%d)'
                            % (test_loss/(batch_idx+1), correct/total, correct, total))

        # Save checkpoint.
        acc = correct/total
        if rank==0 and acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
            best_epoch = epoch

    for epoch in range(start_epoch, start_epoch+args.epoch):
        train(epoch)
        test(epoch)
        scheduler.step()
    # mlflow logging
    if rank == 0:
        log_metric("best_epoch", best_epoch)
        log_metric("best_acc", best_acc)
        log_artifacts("checkpoint")
    # DDP post process
    cleanup()

if __name__ == "__main__":
    mp.spawn(train_and_test,
            args=(args.world_size,),
            nprocs=args.world_size,
            join=True)