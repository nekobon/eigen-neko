# eigen-neko

### Setup:

- (intended for python 3.8)
- Download and unzip files from https://www.kaggle.com/crawford/cat-dataset into a subdirectory "input", removing the "cats" directory (it is a duplicated version of the main folder)
- create a new virtual-env. example: `python -m venv ~/.env-eigeneko` and activate (`. <myvenvdir/bin/activate>`).
- pip install -r requirements.txt
- from the repository, run `pwd > <myvenvdir>/lib/python3.8/site-packages/eigeneko.pth` (Now in IPython, when running `import sys; sys.path`, the repository directory should appear.)
