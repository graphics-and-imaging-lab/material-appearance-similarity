import scipy.io
import torch
from torchvision import transforms
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_embeddings(model, imgs, to_numpy=True):
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
        embs = torch.cat(partial_embs, 0)

    if to_numpy:
        return embs.numpy()
    return embs


if __name__ == '__main__':
    weights_path = '/media/mlagunas/Data/Projects/2019-MATERIAL_SIMILARITY/code/minimal_working/checkpoints_brdf_l4-lab_sim/resnet_brdf_similarity-27_03_2020-23_43/model_best_acc6812.pth.tar'
    folder_path = '/'.join(weights_path.split('/')[:-1])
    imgs_path = 'data/havran1_ennis_298x298_LDR'
    # we will store the obtained feature vectors in this path
    embs_path = folder_path + '/embs_havran_ennis.mat'

    trf = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        # transforms.Lambda(lambda x: x.convert('L')),
        transforms.ToTensor(),
    ])

    model = utils.load_model(weights_path, input_size=3)
    imgs, img_paths = utils.load_imgs(imgs_path, trf_test=trf)
    embs = get_embeddings(model, imgs)
    scipy.io.savemat(embs_path, mdict={'embs': embs, 'img_paths': img_paths})

    print('done')
