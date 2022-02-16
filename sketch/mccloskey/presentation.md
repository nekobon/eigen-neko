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
- We use our aligner to create one row for each file, 64x64 long in grayscale (0-255)
CODE
    all_files = list(utils.Paths.gen_files())
    files = all_files[:n_samples]
    images = [ImageOps.grayscale(aligner.align_one_image(f, 64, 64)) for f in files]
    shape = np.array(images[0]).shape
    X_train = np.array([np.array(im).flatten() for im in images])
END
## SVD
- We get the average image X_mean and subtract it from each image, then take SVD
CODE
    X_mean = X_train.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_train - X_mean, full_matrices=False)
END
- Note: U.shape = (9997, 4096). 9997 images, 4096 pixels each.
-       Vt.Shape = (4096, 4096). Only 4096 principle components
### Get Eigenfaces
- Vt is our principle components, so we expand the first 16 back into images to show 
CODE 
    eigenfaces = np.array([dilate_components(arr) for arr in Vt])
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    ret = utils.plot_portraits(eigenfaces, shape, 4, 4, eigenface_titles)
END
## Reconstruct for Each Percentile
- We then reconstruct each image, using a subset of components with each
- TODO: add more depth here
CODE
    def reconstruct_with_components(image: int | slice, component: int | slice):
        principle_components = U[image, component] * S[component]
        principle_directions = Vt[component]

        return (principle_components @ principle_directions) + X_mean
END
- Then we construct a nice gif for percentile of principle components
CODE
    for n_components in percentiles:
        X_final = reconstruct_with_components(slice(10, 30), slice(n_components))
        fig = utils.plot_portraits(...)
END
# Results - John
_OUTPUTS test
_OUTPUTS simple
_OUTPUTS eyes
_OUTPUTS lstsq_1000
_OUTPUTS lstsq_n10000
# Conclusion - John
- Simple works decently
- Eyes and Least Squares give much better eigenfaces
- The number of components needed to account for all the noise stays consistent
# Future Directions
- Use distance from each eigenvector as labels to train a model that would then classify cats into prospective types (to decompose our own cats)
- Use clustering to cluster groups of cats in the first few PCA dimensions and fine commonalities
- TODO: try not taking out the mean
- TODO: try taking both curves together and look at them
- TODO: 10_000 LSTSQ
- JTODO: sideways cat
- TODO: list sorted files
