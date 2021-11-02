#%%
from core import utils

import matplotlib as plt

files = utils.Paths.gen_files()
file = next(files)

image = utils.parse_image(file.image)

h, w, c = image.shape
# %%
