import os

import PIL
import torch
from torchvision import models, transforms

from model import FTModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
DEFAULT_TRF = transforms.Compose([
    transforms.Resize(size=(256)),
    transforms.CenterCrop(size=(IMG_SIZE)),
    transforms.ToTensor(),
])


def load_model(weights_path=None):
    model, layers_to_remove = models.resnet34(pretrained=True), 1
    model = FTModel(model, layers_to_remove=layers_to_remove, num_features=128, num_classes=100, train_only_fc=False)

    if weights_path is not None:
        print('loading model weights')
        if os.path.isfile(weights_path):
            print(" => loading checkpoint '{}'".format(weights_path))
            checkpoint = torch.load(weights_path)
            model.load_state_dict(checkpoint['state_dict'])
            print(" => loaded checkpoint '{}' (epoch {})"
                  .format(weights_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(weights_path))

    model.to(DEVICE)
    model.eval()

    return model


def load_imgs(imgs_path, trf_test=None):
    def ldr_loader(image_path):
        """loads a path in a PIL image"""
        return PIL.Image.open(image_path)

    if trf_test is None:
        trf_test = DEFAULT_TRF

    # get all the image paths
    img_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(imgs_path) for f in filenames]

    # prepare data structures
    imgs_tensor = torch.zeros(len(img_paths), 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # load all the images
    print('loading images')
    for i, img_path in enumerate(img_paths):
        imgs_tensor[i] = trf_test(ldr_loader(img_path))

    return imgs_tensor, img_paths
