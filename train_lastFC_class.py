import torch
import os
import argparse
import shutil
import numpy as np
import random
import torchvision.transforms.functional as transforms_F

from torch import optim
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms, models
from datetime import datetime
from torchvision.datasets import ImageFolder
import losses
from model import FTModel
from tqdm import tqdm

current_time = datetime.now().strftime("%d_%m_%Y-%H_%M")

parser = argparse.ArgumentParser(description='Material Similarity Training')
parser.add_argument('--train-dir',
                    metavar='DIR', help='path to dataset',
                    default='data/split_dataset')
parser.add_argument('--test-dir',
                    metavar='DIR', help='path to dataset',
                    default='data/havran1_ennis_298x298_LDR')
parser.add_argument('-j', '--workers',
                    default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--epochs',
                    default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--adjust-epoch',
                    nargs='+', default=[20, 40, 60],
                    type=int, help='milestones to adjust the learning rate')
parser.add_argument('--num-classes', default=100, type=int,
                    help='number of classes in the problem')
parser.add_argument('--emb-size',
                    default=128, type=int, help='size of the embedding')
parser.add_argument('-b', '--batch-size',
                    default=20, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate',
                    default=5e-2, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay',
                    default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 5e-5)',
                    dest='weight_decay')
parser.add_argument('--betas',
                    nargs='+', default=[0.9, 0.999], type=float,
                    help='beta values for ADAM')
parser.add_argument('--margin',
                    default=0.3, type=float,
                    help='triplet loss margin')
parser.add_argument('--checkpoint-folder',
                    default='./checkpoints_lastFC_class',
                    type=str, help='folder to store the trained models')
parser.add_argument('--model-name',
                    default='resnet_similarity', type=str,
                    help='name given to the model')
parser.add_argument('--resume',
                    default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate',
                    dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed',
                    default=2851, type=int,
                    help='seed for initializing training. ')


class AverageMeter(object):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/single.py
    Computes and stores the average and current value
    """

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


class RandomResize(object):
    def __init__(self, low, high, interpolation=Image.BILINEAR):
        self.low = low
        self.high = high
        self.interpolation = interpolation

    def __call__(self, img):
        size = np.random.randint(self.low, self.high)
        return transforms_F.resize(img, size, self.interpolation)


def accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions for the specified
        values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def iterate_model(loader, epoch, is_train=True):
    def update_progress_bar(progress_bar, losses, acc_t1, acc_t5):
        description = '[' + str(epoch) + '-'
        description += '%s]' % 'train' if is_train else 'val'
        description += ' Loss: %.4f/ %.4f (AVG)' % (losses.val, losses.avg)
        description += ' Acc (t1) %2.2f/ %2.2f (AVG)' % (acc_t1.val, acc_t1.avg)
        description += ' Acc (t5) %2.2f/ %2.2f (AVG)' % (acc_t5.val, acc_t5.avg)
        progress_bar.set_description(description)

    global model
    global criterion
    global optimizer

    # keep track of the loss value
    losses = AverageMeter()
    acc_t1 = AverageMeter()
    acc_t5 = AverageMeter()

    progress_bar = tqdm(loader, total=len(loader))
    for imgs, targets in progress_bar:
        with torch.set_grad_enabled(is_train):
            imgs = imgs.to(device, dtype)
            targets = targets.to(device, dtype)

            # forward through the model and compute error
            preds, _ = model(imgs)
            loss = criterion(preds, targets)
            losses.update(loss.item(), imgs.size(0))

            if is_train:
                # compute gradient and update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t1, t5 = accuracy(preds, targets, topk=(1, 5))
            acc_t1.update(t1, len(targets))
            acc_t5.update(t5, len(targets))

            update_progress_bar(progress_bar, losses, acc_t1, acc_t5)

    return losses.avg


def get_transforms():
    # set image transforms
    trf_train = transforms.Compose([
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.CenterCrop(size=384),
        RandomResize(low=256, high=384),
        transforms.RandomCrop(size=224),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    trf_test = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
    ])
    return trf_train, trf_test


def get_dataloaders(trf_train, trf_test):
    global args

    def _init_loader_(worker_id):
        np.random.seed(args.seed + worker_id)

    loader_args = {
        'batch_size': args.batch_size,
        'num_workers': args.workers,
        'pin_memory': True,
        'worker_init_fn': _init_loader_,
    }

    loader_train = DataLoader(
        dataset=ImageFolder(
            root=os.path.join(args.train_dir, 'train'),
            transform=trf_train,
        ),
        shuffle=True,
        drop_last=True,
        **loader_args
    )
    loader_val = DataLoader(
        dataset=ImageFolder(
            root=os.path.join(args.train_dir, 'val'),
            transform=trf_test,
        ),
        shuffle=True,
        **loader_args
    )

    loader_test = DataLoader(
        dataset=ImageFolder(
            root=os.path.join(args.train_dir, 'test'),
            transform=trf_test,
        ),
        shuffle=True,
        **loader_args
    )

    return loader_train, loader_val, loader_test


def save_checkpoint(state, is_best, folder, model_name='checkpoint', ):
    """
    if the current state is the best it saves the pytorch model
    in folder with name filename
    """
    path = os.path.join(folder, model_name)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'model')

    torch.save(state, path + '.pth.tar')
    if is_best:
        shutil.copyfile(path + '.pth.tar', path + '_best.pth.tar')


if __name__ == '__main__':

    # get input arguments
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # set device and dtype
    dtype = torch.float
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        # comment this if you want reproducibility
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # # # this might affect performance but allows reproducibility
        # torch.backends.cudnn.enabled = False
        # torch.backends.cudnn.deterministic = True

    # define dataset
    trf_train, trf_test = get_transforms()
    loader_train, loader_val, loader_test = get_dataloaders(trf_train, trf_test)

    # create model
    model = FTModel(
        models.resnet34(pretrained=True),
        layers_to_remove=1,
        train_only_fc=True,
        num_features=args.emb_size,
        num_classes=args.num_classes,
    )
    model = model.to(device, dtype)

    args.resume = 'data/resnet_similarity_best.pth.tar'
    if args.resume is not None:
        if os.path.isfile(args.resume):
            tqdm.write("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            tqdm.write("=> loaded checkpoint '{}' (epoch {})"
                       .format(args.resume, checkpoint['epoch']))
        else:
            tqdm.write("=> no checkpoint found at '{}'".format(args.resume))

    # define loss function
    criterion_tl = losses.TripletLossHuman(
        margin=args.margin,
        unit_norm=True,
        device=device,
        seed=args.seed
    )
    criterion = losses.CrossEntropyLabelSmooth(
        num_classes=args.num_classes,
        device=device,

    )

    # define optimizer
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     betas=args.betas,
    #     weight_decay=args.weight_decay,
    #     lr=args.lr,
    #     amsgrad=True
    # )
    optimizer = optim.SGD(
        model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.lr,
        momentum=0.9,
        nesterov=True
    )

    # define LR scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.adjust_epoch,
        gamma=0.1,
    )

    # set a high value for the error
    best_agreement = 0

    if args.evaluate:
        model = model.eval()
        iterate_model(loader_train, 0, is_train=True)

    else:
        # start training and evaluation loop
        for epoch in range(args.start_epoch + 1, args.epochs + 1):
            # train step
            model = model.train()
            iterate_model(loader_train, epoch, is_train=True)
            lr_scheduler.step()

            # evaluation step
            model = model.eval()
            current_metric = iterate_model(loader_val, epoch, is_train=False)

            # checkpoint model if it is the best until now
            is_best = current_metric > best_agreement
            best_agreement = max(current_metric, best_agreement)
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_agreement': best_agreement,
                    'optimizer': optimizer.state_dict(),
                },
                is_best, folder=args.checkpoint_folder,
                model_name=args.model_name + '-' + current_time
            )

        current_metric = iterate_model(loader_test, -1, is_train=False)
