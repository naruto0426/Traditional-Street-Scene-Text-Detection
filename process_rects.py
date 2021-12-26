import glob
from natsort import natsorted
import re
import cv2
from tqdm import tqdm
import json
def write_bd_to_file(gt,is_line=False):
    x1,y1,x2,y2,x3,y3,x4,y4 = gt['points']
    text = gt['transcription']
    if gt['ignore']:
        id = 2
    elif len(text)==1:
        id = 1
    else:
        id = 0
    if not is_line or len(text)>1:
        min_x = min(x1,x2,x3,x4)
        max_x = max(x1,x2,x3,x4)
        min_y = min(y1,y2,y3,y4)
        max_y = max(y1,y2,y3,y4)
        c_x = (min_x+max_x)/2/W
        c_y = (min_y+max_y)/2/H
        w = (max_x-min_x)/W
        h = (max_y-min_y)/H
        if w>0 and h>0:
            print(id,c_x,c_y,w,h,file=out_file)
for path in tqdm(natsorted(glob.glob("rects/img/*.jpg"))):
    img = cv2.imread(path)
    H,W = img.shape[:2]
    base = path[4:-4]
    with open(f"rects/gt/{base}.json","r") as f:
        data = json.load(f)
    with open(f"rects/img/{base}.txt","w+") as out_file:
        for gt in data['chars']:
            write_bd_to_file(gt)
        for gt in (data['chars']+data['lines']):
            write_bd_to_file(gt,is_line=True)

