# Outline
![my image](images/00000100_014.jpg)
# Human Face Recognition - John/Yu
# Cat Dataset - Yu
## What points mean
# Project Flow - Yu
# Dataset Cleaning - Yu
## Simple Cat Aligner
## Eye Based Cat Aligner
## LSTSQ Cat Aligner
# PCA Decomposition - John 
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
