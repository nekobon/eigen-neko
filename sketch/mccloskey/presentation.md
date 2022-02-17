# Cat Presentation
Yu Tomita and John McCloskey
![my image](images/00000100_014.jpg)
# Outline
- Cat Dataset Overview
- Program Flow
- Data Cleaning (Cat Aligners)
- Singular Value Decomposition
- Results
- Conclusion / Future Directions
NOTEBOOK notebooks/yu_notebook.ipynb
# Singular Value Decomposition
NOTEBOOK_SUBSLIDE notebooks/john_notebook.ipynb
# Results
_OUTPUTS simple_100
## ![my image](images/simple_100/gif_subimages/1_components.png)
## ![my image](images/simple_100/gif_subimages/2_components.png)
## ![my image](images/simple_100/gif_subimages/3_components.png)
_OUTPUTS simple
## ![my image](images/simple/gif_subimages/1_components.png)
## ![my image](images/simple/gif_subimages/3_components.png)
## ![my image](images/simple/gif_subimages/5_components.png)
_OUTPUTS eyes
## ![my image](images/eyes/gif_subimages/1_components.png)
## ![my image](images/eyes/gif_subimages/2_components.png)
## ![my image](images/eyes/gif_subimages/4_components.png)
_OUTPUTS lstsq
## ![my image](images/lstsq/gif_subimages/1_components.png)
## ![my image](images/lstsq/gif_subimages/3_components.png)
## ![my image](images/lstsq/gif_subimages/5_components.png)
# Conclusion
- Simple works decently
- Eyes and Least Squares give much better eigenfaces
- The number of components needed to account for all the noise stays consistent
# Future Directions
- Use distance from each eigenvector as labels to train a model that would then classify cats into prospective types (to decompose our own cats)
- Use clustering to cluster groups of cats in the first few PCA dimensions and fine commonalities
