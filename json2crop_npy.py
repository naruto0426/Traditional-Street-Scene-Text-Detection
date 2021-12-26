import json
import os
import glob
from pathlib import Path
import cv2
import re
from tqdm import tqdm
import numpy as np
os.makedirs("../crop_img",exist_ok=True)
root = "ai_data/train/json"
save_root = "ai_data/train/img"
json_paths = glob.glob(f"{root}/*.json")
classes = []
char_size = 64
i = 0
crop_txt = open("../crop_img/crop_img.txt","w+")
for path in tqdm(json_paths):
    with open(path,"r") as f:
        data = json.load(f)
    img_path = data['imagePath']
    w_img = int(data['imageWidth'])
    h_img = int(data['imageHeight'])
    base = Path(path).stem
    out_file = open(f"{save_root}/{base}.txt","w+")
    img = cv2.imread(f"{save_root}/{img_path}")
    for shape in data['shapes']:
        id = shape['group_id']
        points = shape['points']
        (x_lu,y_lu),(x_ru,y_ru),(x_rd,y_rd),(x_ld,y_ld) = points
        min_x = max(min(x_rd,x_ld,x_ru,x_lu),0)
        max_x = min(max(x_rd,x_ld,x_ru,x_lu),w_img)
        min_y = max(min(y_rd,y_ld,y_ru,y_lu),0)
        max_y = min(max(y_rd,y_ld,y_ru,y_lu),h_img)
        label = shape['label']
        img_crop = img[min_y:max_y,min_x:max_x]
        if (len(label)==1 and len(re.findall(r'[\u4e00-\u9fff]+', label))>0) or id==255:
            if 0 not in img_crop.shape:
                img_char = cv2.resize(img_crop, (char_size, char_size), interpolation=cv2.INTER_AREA)
                save_name = f"crop_img/{i}.jpg"
                cv2.imwrite("../"+save_name,img_char)
                label = label if id!=255 else '#'
                print(save_name+","+label,file=crop_txt)
                i+=1
            else:
                print(img_crop.shape)
                print(label)
                print(img_path)
        #if id != 255: #don't care
        if id==255:
            id_new = 2
        elif len(label)==1:
            id_new = 1
        else:
            id_new = 0
        if id_new not in classes:
            classes += [id_new]
        if 0 not in img_crop.shape:
            c_x = (min_x+max_x)/2/w_img
            c_y = (min_y+max_y)/2/h_img
            w = (max_x-min_x)/w_img
            h = (max_y-min_y)/h_img
            print(id_new,c_x,c_y,w,h,file=out_file)
    out_file.close()
crop_txt.close()
print("len for class",len(classes))
print(classes)
