import numpy as np
import cv2
import os
import pathlib
import time

cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
cat_ext_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')

SF=1.05 #1.3  # try different values of scale factor like 1.05, 1.3, etc
N=2 # try different values of minimum neighbours like 3,4,5,6

# 6: 49.82% 51.41%
# 4: 58.44% 59.20%
# 2: 73.39% 72.92%

root_dir = pathlib.Path('/home/tomita/eigen-neko/')

input_dir = root_dir / 'input' 
output_dir = root_dir / 'output'

num_total = 0
num_cats = 0
num_cats_ext = 0

def processImage(fp):
    global num_total, num_cats, num_cats_ext
    # read the image
    img = cv2.imread(str(fp))
    # convery to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # this function returns tuple rectangle starting coordinates x,y, width, height
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
    #print(cats) # one sample value is [[268 147 234 234]]
    cats_ext = cat_ext_cascade.detectMultiScale(gray, scaleFactor=SF, minNeighbors=N)
    #print(cats_ext)

    num_total += 1
    has_cats = bool(len(cats))
    num_cats += has_cats 
    has_cats_ext = bool(len(cats_ext))
    num_cats_ext += has_cats_ext
    print(f'{100 * num_cats / num_total:.2f}%', f'{100 * (num_cats_ext / num_total):.2f}%')
    
    # draw a blue rectangle on the image
    for (x,y,w,h) in cats:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)       
    # draw a green rectangle on the image 
    for (x,y,w,h) in cats_ext:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    # save the image to a file
    rel_fp = fp.relative_to(input_dir)
    status = cv2.imwrite(str('/home/tomita/eigen-neko/output' / rel_fp), img)
    if not status:
        raise RuntimeError(fp)
    

for fp in input_dir.glob('CAT_00/*.jpg'):
    print(fp)
    processImage(fp)
