import json
import torch
import utils
import scipy.io

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def euc_dist(x, y): return (((x - y) ** 2).sum()).sqrt()


def mse_dist(x, y): return ((x - y) ** 2).mean()


def manhattan_dist(x, y): return ((x - y).abs()).sum()


if __name__ == '__main__':

    path_to_users_answers = 'data/answers_processed.json'
    # embs_path = 'data/embs_similarity_efficientnet_havran_ennis.mat'
    embs_path = 'data/embs_similarity_resnet_havran_ennis.mat'

    with open(path_to_users_answers) as f:
        user_data = json.load(f)

    users_answers = torch.LongTensor(user_data['answers_diff'])
    users_agreement = torch.LongTensor(user_data['agreement_diff'])

    embs = torch.tensor(scipy.io.loadmat(embs_path)['embs'])

    dist_to_apply = [euc_dist, mse_dist, manhattan_dist]
    res = [0] * len(dist_to_apply)
    for agr, imgs_ix in zip(users_agreement, users_answers):
        r, a, n = imgs_ix
        for i, dist in enumerate(dist_to_apply):
            positive_dist = dist(embs[r], embs[a])
            negative_dist = dist(embs[r], embs[n])
            if negative_dist > positive_dist: res[i] += 1

    for r in res:
        print(r / len(users_agreement))
