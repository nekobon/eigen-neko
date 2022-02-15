# Cat Presentation
Yu Tomita and John McCloskey
![my image](images/00000100_014.jpg)
# Outline
# Human Face Recognition - John/Yu
# Cat Dataset - Yu
## What points mean
# Project Flow - Yu
# Dataset Cleaning - Yu
## Simple Cat Aligner
## Eye Based Cat Aligner
## LSTSQ Cat Aligner
# PCA Decomposition - John 
## Construct Training Data
CODE
    all_files = list(utils.Paths.gen_files())
    files = all_files[:n_samples]
    images = [ImageOps.grayscale(aligner.align_one_image(f, 64, 64)) for f in files]
    shape = np.array(images[0]).shape
    X_train = np.array([np.array(im).flatten() for im in images])
END
## Take mean, do SVD
CODE
    X_mean = X_train.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_train - X_mean, full_matrices=False)
END
## Get Eigenfaces
CODE 
    eigenfaces = np.array([dilate_components(arr) for arr in Vt])
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    ret = utils.plot_portraits(eigenfaces, shape, 4, 4, eigenface_titles)
END
## Reconstruct for Each Percentile
CODE
    def reconstruct_with_components(image: int | slice, component: int | slice):
        principle_compoents = U[image, component] * S[component]
        principle_directions = Vt[component]

        return (principle_compoents @ principle_directions) + X_mean
END
CODE
    for n_components in percentiles:
        X_final = reconstruct_with_components(slice(10, 30), slice(n_components))
        fig = utils.plot_portraits(...)
END
# Results - John
_OUTPUTS simple_n10000
_OUTPUTS eyes_n10000
_OUTPUTS lstsq_n1000
_OUTPUTS lstsq_n10000
# Conclusion - John
# This is a big title
## This is my subtitle 
- My idea
CODE
class PCAResult(tp.NamedTuple):
    mean: np.ndarray
    centered_data: np.ndarray
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray


N_SAMPLES = 500
# """It helps visualising the portraits from the dataset."""
def plot_portraits(images, titles, shape, n_row, n_col):
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape(shape), cmap="gray")
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
    plt.show()
END
# Hopefully jupyter should follow
