import numpy as np
import scipy.io
import torch
import umap
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage import transform

np.random.seed(123)
torch.manual_seed(123)


def _imscatter(x, y, image, color=None, ax=None, zoom=1.):
    """ Auxiliary function to plot an image in the location [x, y]
        image should be an np.array in the form H*W*3 for RGB
    """
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
        size = min(image.shape[0], image.shape[1])
        image = transform.resize(image[:size, :size], (256, 256))
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        edgecolor = dict(boxstyle='round,pad=0.05',
                         edgecolor=color, lw=4) \
            if color is not None else None
        ab = AnnotationBbox(im, (x0, y0),
                            xycoords='data',
                            frameon=False,
                            bboxprops=edgecolor,
                            )
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


if __name__ == '__main__':
    embs_path = '/media/mlagunas/Data/Projects/2019-MATERIAL_SIMILARITY/code/minimal_working/checkpoints_base_class_pretrained_ours/resnet_similarity-16_03_2020-11_37/embs_havran_ennis.mat'
    do_unit_norm = True

    mat_file = scipy.io.loadmat(embs_path)
    embs = torch.tensor(mat_file['embs'])

    if do_unit_norm:
        embs /= embs.norm(p=2, dim=1, keepdim=True)
        embs = embs.numpy()

    img_paths = [str(elem).strip() for elem in mat_file['img_paths']]

    # get umap from the embeddings
    umap_fit = umap.UMAP(n_neighbors=5,
                         init='spectral',
                         min_dist=3,
                         spread=10,
                         metric='l2',
                         transform_seed=123)
    umap_emb = umap_fit.fit_transform(embs)
    np.save('data/class_umap.npy', umap_emb)
    # plot each point of the umap as its corresponding image
    fig = plt.figure()
    ax = fig.gca()
    for i, point in enumerate(umap_emb):
        _imscatter(point[0], point[1], img_paths[i], zoom=0.12, ax=ax)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("data/classification_umap.pdf")

    # plt.gca().invert_yaxis()
    # plt.gca().invert_xaxis()
    plt.show()
