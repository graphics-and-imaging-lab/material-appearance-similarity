import torch
import torch.nn as nn
import json
import warnings
import utils
from get_embs import get_embeddings
import random
import numpy as np


class TripletLossHuman(nn.Module):
    def __init__(self, margin=0.3, unit_norm=False, device=None, seed=None):
        super(TripletLossHuman, self).__init__()

        # set seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # user answers path
        mturk_data_train = 'data/answers_processed_train.json'
        mturk_data_test = 'data/answers_processed_test.json'

        with open(mturk_data_train) as f:
            mturk_data_train = json.load(f)
        with open(mturk_data_test) as f:
            mturk_data_test = json.load(f)

        # set loss variables
        self.margin = margin
        self.unit_norm = unit_norm

        # triplet loss function used to model user answers
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

        # get user answers and agreement (note that we only take different
        # agreements because an agreement of 2-2 will not be suitable for
        # training a triplet metric)
        self.user_answers_train = mturk_data_train['answers_diff']
        self.user_agreement_train = mturk_data_train['agreement_diff']
        self.user_answers_test = mturk_data_test['answers_diff']
        self.user_agreement_test = mturk_data_test['agreement_diff']

        self.user_answers_train = torch.tensor(self.user_answers_train)
        self.user_agreement_train = torch.tensor(self.user_agreement_train)
        self.user_answers_test = torch.tensor(self.user_answers_test)
        self.user_agreement_test = torch.tensor(self.user_agreement_test)

        self.user_answers_train = self.user_answers_train.to(device).long()
        self.user_agreement_train = self.user_agreement_train.to(device).long()
        self.user_answers_test = self.user_answers_test.to(device).long()
        self.user_agreement_test = self.user_agreement_test.to(device).long()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        if self.unit_norm:
            inputs = inputs / inputs.norm(dim=1, keepdim=True)

        # Compute pairwise distances
        dist = utils.pairwise_dist(inputs)

        # move answers to correct device
        targets = targets.long()

        # get triplets answered by humans with representation in the batch
        is_there = torch.zeros_like(self.user_answers_train)
        for target in torch.unique(targets):
            is_there = is_there + (self.user_answers_train == target).long()
        idx_triplets = (is_there.sum(dim=1) == 3).nonzero()
        if len(idx_triplets) == 0:
            warnings.warn('\nZero sampled triplets. '
                          'Consider increasing your batch size')
            return torch.zeros(1, requires_grad=True).to(targets.device)

        dist_ap = torch.zeros(len(idx_triplets))
        dist_an = torch.zeros(len(idx_triplets))
        # target_agreement = torch.zeros(len(idx_triplets))

        dist_ap = dist_ap.to(inputs.device, inputs.dtype)
        dist_an = dist_an.to(inputs.device, inputs.dtype)
        # target_agreement = target_agreement.to(inputs.device, inputs.dtype)

        for i, idx in enumerate(idx_triplets):
            # get index of the triplet according to the given targets
            target_triplet = self.user_answers_train[idx].squeeze()
            triplet_idx = \
                (targets.view(1, -1) == target_triplet.view(-1, 1)).nonzero()

            # get single elements for each class.
            ix0 = triplet_idx[(triplet_idx[:, 0] == 0).nonzero()].squeeze()
            ix1 = triplet_idx[(triplet_idx[:, 0] == 1).nonzero()].squeeze()
            ix2 = triplet_idx[(triplet_idx[:, 0] == 2).nonzero()].squeeze()

            # At the end, the classes repeated have the same feature vector
            if len(ix0.shape) > 1: ix0 = ix0[0]
            if len(ix1.shape) > 1: ix1 = ix1[0]
            if len(ix2.shape) > 1: ix2 = ix2[0]

            # get distances and agreement
            dist_ap[i] = dist[ix0[1], ix1[1]]
            dist_an[i] = dist[ix0[1], ix2[1]]
            # target_agreement[i] = (self.user_agreement_train[idx, 0] /
            #                        self.user_agreement_train[idx].sum())

        # Compute ranking hinge loss
        loss_human = self.ranking_loss(dist_ap, dist_an,
                                       torch.zeros_like(dist_ap))

        # move from distances to probabilities
        s_ap = 1 / (dist_ap + 1)
        s_an = 1 / (dist_an + 1)

        # compute perplexity
        p_ap = s_ap / (s_ap + s_an)
        loss_perplexity = (-torch.log(p_ap + 1e-8)).mean()

        # get total loss and return
        loss = loss_human + loss_perplexity
        return loss

    def get_majority_accuracy(self, mturk_images, model, train=False,
                              unit_norm=False):
        """ test users answers against the response of the trained model.
            if train=true it tests on the train answers if test=true it
            tests in the subset of answers reserved for testing.
        """
        embs = get_embeddings(model, mturk_images, to_numpy=False)
        if unit_norm:
            embs = embs / embs.norm(dim=1, keepdim=True)

        dist = utils.pairwise_dist(embs)

        agreement_count = 0
        if train:
            iterator = zip(self.user_agreement_train, self.user_answers_train)
            total = len(self.user_answers_train)
        else:
            iterator = zip(self.user_agreement_test, self.user_answers_test)
            total = len(self.user_answers_test)

        for agr, imgs_ix in iterator:
            r, a, n = imgs_ix
            positive_dist = dist[r, a]
            negative_dist = dist[r, n]
            if positive_dist < negative_dist: agreement_count += 1

        return agreement_count / total


class TripletLossBrdfSim(nn.Module):
    def __init__(self,
                 brdf_metric_dist,
                 margin=0.3,
                 unit_norm=False,
                 device=None,
                 seed=None):
        super(TripletLossBrdfSim, self).__init__()

        # set seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # set loss variables
        self.margin = margin
        self.unit_norm = unit_norm
        self.brdf_metric_dist = brdf_metric_dist

        # triplet loss function used to model user answers
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        if self.unit_norm:
            inputs = inputs / inputs.norm(dim=1, keepdim=True)

        # Compute pairwise distances
        dist = utils.pairwise_dist(inputs)

        # move answers to correct device
        targets = targets.long()

        # For each anchor, find the hardest positive and negative
        batch_size = inputs.size(0)

        # sample triplets
        dist_ap = torch.zeros(batch_size).to(inputs.device, inputs.dtype)
        dist_an = torch.zeros(batch_size).to(inputs.device, inputs.dtype)
        for ix in range(batch_size):
            # get two random indexes
            ixs = torch.randint(low=0, high=batch_size - 1, size=(2,))
            ixs = ixs.to(inputs.device, torch.long)
            while ix in ixs:
                ixs = torch.randint(low=0, high=batch_size - 1, size=(2,))
                ixs = ixs.to(inputs.device, torch.long)

            # get their distances to the anchor according to the BRDF metric
            metric_dist_1 = self.brdf_metric_dist[targets[ix], targets[ixs[0]]]
            metric_dist_2 = self.brdf_metric_dist[targets[ix], targets[ixs[1]]]

            # get their model distances and take into account
            # the positive/ negative image from the BRDF metric
            if metric_dist_1 < metric_dist_2:
                dist_ap[ix] = dist[ix, ixs[0]]
                dist_an[ix] = dist[ix, ixs[1]]
            else:
                dist_ap[ix] = dist[ix, ixs[1]]
                dist_an[ix] = dist[ix, ixs[0]]

        loss_metric = (dist_ap - dist_an + self.margin).clamp(min=0.).mean()
        # Compute ranking hinge loss
        # zeros = torch.zeros_like(dist_ap)
        # loss_human = self.ranking_loss(dist_ap, dist_an, zeros)

        # move from distances to probabilities
        s_ap = 1 / (dist_ap + 1)
        s_an = 1 / (dist_an + 1)

        # compute perplexity
        p_ap = s_ap / (s_ap + s_an)
        loss_perplexity = (-torch.log(p_ap + 1e-8)).mean()

        # get total loss and return
        loss = loss_metric  # + loss_perplexity
        return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with a label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision.
    CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self,
                 num_classes,
                 epsilon=0.1,
                 device='cpu',
                 dtype=torch.float):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.device = device
        self.dtype = dtype
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        zeros = torch.zeros(log_probs.size()).to(self.device, torch.long)
        targets_ = targets.unsqueeze(1).detach().to(self.device, torch.long)
        targets = zeros.scatter_(1, targets_, 1)
        targets = targets.to(self.device, self.dtype)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
