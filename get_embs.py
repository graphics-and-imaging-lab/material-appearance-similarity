import scipy.io
import torch

import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_embeddings(model, imgs):
    print('getting embeddings')

    # avoid computing gradients
    with torch.set_grad_enabled(False):
        # list to store the embeddings
        partial_embs = []

        # number of images and embeddings forwarded at once
        n_steps = 10
        offset = (len(imgs) // n_steps)

        # loop to obtain the embeddings
        i, idx_start, idx_end = 0, 0, 0
        while idx_end < len(imgs):
            idx_start = offset * i
            idx_end = min(offset * (i + 1), len(imgs))

            # forward through the CNN and get the embeddings
            _, embs = model.forward(imgs[idx_start:idx_end])
            partial_embs.append(embs.detach().cpu())

            # update index
            i += 1

        # move it to numpy and return
        embs = torch.cat(partial_embs, 0).numpy()
    return embs


if __name__ == '__main__':

    # get weights path to load in the model, path to store embeddings and if the
    # model is Efficient Net.
    # weights, embs_path, is_efficientnet = \
    #     'data/resnet_similarity_best.pth.tar', \
    #     'data/resnet_similarity_embs_havran_ennis', \
    #     False
    # weights, embs_path, is_efficientnet = \
    #     'data/resnet_classification_best.pth.tar', \
    #     'data/resnet_classification_embs_havran_ennis', \
    #     False
    weights_path, embs_path, is_efficientnet = \
        'data/efficientnet_similarity_best.pth.tar', \
        'data/efficientnet_similarity_embs_havran_ennis', \
        True

    # path to the images we will extract their feature vectors
    imgs_path = 'data/havran1_ennis_298x298_LDR'

    if is_efficientnet:
        model = utils.load_model_ef(weights_path)
    else:
        model = utils.load_model(weights_path)

    imgs, img_paths = utils.load_imgs(imgs_path)
    embs = get_embeddings(model, imgs)

    scipy.io.savemat(embs_path, mdict={'embs': embs, 'img_paths': img_paths})

    print('done')
