import jsonlines
import hanzidentifier
import numpy as np
import cv2
import os
from tqdm import tqdm
os.makedirs('ctw',exist_ok=True)
chars = open("chars_list.txt",'r').read().split('\n')
chars = [char for char in chars if char!='']
with jsonlines.open('ctw-annotations/train.jsonl') as f:
    c=list(f)
with open("chars_list_with_ctw_and_new_ver2.txt","r+") as text_file:
    ts = text_file.read().split("\n")
texts = []
dataset_txt = open("ctw/dataset.txt","w+")
for i in tqdm(range(len(c))):
    x = c[i]['annotations']
    f_name = c[i]['file_name']
    f_id = c[i]['image_id']
    img = cv2.imread('ctw_dataset/img/'+f_name)
    yolo_gt_path = "ctw_dataset/img/"+f_name[:-4]+".txt"
    n = 0
    f = open(yolo_gt_path,"w+")
    for j in range(len(x)):
        y = x[j]
        bounding_min_x = c[i]['width']
        bounding_min_y = c[i]['height']
        bounding_max_x = 0
        bounding_max_y = 0
        have_obj_count = 0
        for k in range(len(y)):
            z = y[k]
            text = z['text']
            if z["is_chinese"]: #hanzidentifier.is_traditional(text) and z["is_chinese"]:
                texts += [text]
                if text not in chars:
                    chars += [text]
                min_x,min_y,w,h = z['adjusted_bbox']
                min_x = int(max(min_x,0))
                max_x = int(min_x+w)
                min_y = int(max(min_y,0))
                max_y = int(min_y+h)
                bounding_min_x = min(bounding_min_x,min_x)
                bounding_max_x = max(bounding_max_x,max_x)
                bounding_min_y = min(bounding_min_y,min_y)
                bounding_max_y = max(bounding_max_y,max_y)
                save_name = f"ctw/crop_{f_id}_{n}.jpg"
                if img[min_y:max_y,min_x:max_x].size == 0:
                    print("empty",f_name,[min_y,max_y,min_x,max_x])
                else:
                    have_obj_count += 1
                    if text in ts:
                        print(f"{save_name},{text}",file=dataset_txt)
                        #print(min_x,min_y,max_x,max_y)
                        cv2.imwrite(save_name,img[min_y:max_y,min_x:max_x])
                    if len(text)==1:
                        cls_id = 1
                    else:
                        cls_id = 0
                    cx = (min_x+w/2)/c[i]['width']
                    cy = (min_y+h/2)/c[i]['height']
                    w_norm = w/c[i]['width']
                    h_norm = h/c[i]['height']
                    print(cls_id,cx,cy,w_norm,h_norm,file=f)
                    n += 1
        if have_obj_count > 1:
            cx = (bounding_min_x+bounding_max_x)/2/c[i]['width']
            cy = (bounding_min_y+bounding_max_y)/2/c[i]['height']
            w_norm = (bounding_max_x-bounding_min_x)/c[i]['width']
            h_norm = (bounding_max_y-bounding_min_y)/c[i]['height']
            print(0,cx,cy,w_norm,h_norm,file=f)

    f.close()
dataset_txt.close()
text_u = np.unique(np.array(texts))
print(len(text_u))
print(text_u)
with open("chars_list_with_ctw.txt","w+") as f:
    for char in chars:
        print(char,file=f)
