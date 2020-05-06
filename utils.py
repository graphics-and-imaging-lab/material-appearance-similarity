import os
import PIL
import torch
from torchvision import models, transforms
from tqdm import tqdm
from model import FTModel
import numpy as np
from itertools import combinations

import numpy as np
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
DEFAULT_TRF = transforms.Compose([
    transforms.Resize(size=(256)),
    transforms.CenterCrop(size=(IMG_SIZE)),
    transforms.ToTensor(),
])


class FunctionNegativeTripletSelector:
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn,
                 cpu=DEVICE == torch.device('cuda')):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pairwise_dist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().long().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(
                combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[
                anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives,
                                                    ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(
                    np.array([anchor_positive[0]])), torch.LongTensor(
                    negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append(
                        [anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append(
                [anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def semihard_negative(loss_values, margin):
    semihard_negatives = \
        np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(
        semihard_negatives) > 0 else None


def SemihardNegativeTripletSelector(margin, cpu=DEVICE == torch.device('cpu')):
    return FunctionNegativeTripletSelector(
        margin=margin,
        negative_selection_fn=lambda x: semihard_negative(x, margin),
        cpu=cpu)


def load_model(weights_path=None, input_size=3):
    """ Load model with IMAGENET weights if other pretrained weights
    are not given
    """
    model, layers_to_remove = models.resnet34(
        pretrained=weights_path is None), 1
    model = FTModel(model,
                    layers_to_remove=layers_to_remove,
                    input_size=input_size,
                    num_features=128,
                    num_classes=100,
                    train_only_fc=False)

    if weights_path is not None:
        print('loading model weights')
        if os.path.isfile(weights_path):
            print(" => loading checkpoint '{}'".format(weights_path))
            checkpoint = torch.load(weights_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['state_dict'])
            print(" => loaded checkpoint '{}' (epoch {})"
                  .format(weights_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(weights_path))

    model.to(DEVICE)
    model.eval()

    return model


def pil_loader(image_path):
    """loads a path in a PIL image
    """
    return PIL.Image.open(image_path).convert('RGB')


def load_imgs(imgs_path, trf_test=None):
    """ Loads imgs from a folder. If no transforms are given it just
        resizes and performs a center crop of the image.
    """

    if trf_test is None:
        trf_test = DEFAULT_TRF

    # get all the image paths
    img_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(imgs_path)
                 for f in filenames]
    img_paths.sort()

    # load all the images
    imgs_tensor = [None] * len(img_paths)
    tqdm.write('loading images')
    for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        imgs_tensor[i] = (trf_test(pil_loader(img_path)))
    imgs_tensor = torch.stack(imgs_tensor).to(DEVICE)
    return imgs_tensor, img_paths


def pairwise_dist(x):
    """ Computes pairwise distances between features
        x : torch tensor with shape Batch x n_features
    """
    n = x.size(0)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, x, x.t())
    dist = dist.clamp(min=1e-12).sqrt()  # numerical stability
    return dist
