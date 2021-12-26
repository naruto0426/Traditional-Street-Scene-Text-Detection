#pip install Shapely
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch,cv2
import ssl
import sys
from pathlib import Path
from shapely.geometry import Polygon
import glob
from tqdm import tqdm
from natsort import natsorted
import csv
import copy
import math
file = Path(__file__).resolve()
parent, top = file.parent, file.parents[3]
sys.path.append(str(parent)+"/yolov5")
sys.path.append(str(parent)+"/CRNN_Chinese_Characters_Rec")
from CRNN_Chinese_Characters_Rec.demo import *
from CRNN_Chinese_Characters_Rec import EasyOCR
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from PIL import ImageFont, ImageDraw, Image
import easyocr
import re
import torch.nn.functional as F
#import deep_text_demo
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang='en')
##################### plot Saliency map ###############################
def is_english_or_num(text):
    return ((text!="###") and text and (re.sub(r"[0-9a-zA-Z]*[^\w]*",'',text)==""))
def normalize(image):
    return (image - image.min()) / (image.max() - image.min())
    # return torch.log(image)/torch.log(image.max())

def compute_saliency_maps(x, y=None, model=None):
    model.train()

    x = x.cuda()

    # we want the gradient of the input x
    x.requires_grad_()

    y_pred = model(x).cpu()
    batch_size = 1
    preds_size = torch.IntTensor([y_pred.size(0)] * batch_size)

    if y is None:
        _, preds = y_pred.max(2)
        y = preds.transpose(1, 0).contiguous().view(-1)
        print('x',x.shape,'y',y,'data',y.data)
        preds_size = torch.IntTensor([preds.size(0)])
        print(converter.decode(y.data, preds_size, raw=False))
    loss_func = torch.nn.CTCLoss()
    loss = loss_func(y_pred, y,preds_size,preds_size)
    loss.backward()

    # saliencies = x.grad.abs().detach().cpu()
    saliencies, _ = torch.max(x.grad.data.abs().detach().cpu(),dim=1)
    model.eval()

    # We need to normalize each image, because their gradients might vary in scale, but we only care about the relation in each image
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies
def plot_map(img,label=None,model=None):
    if label:
        label_tensor = torch.tensor(label)
    else:
        label_tensor = None
    img_h, img_w = img.shape
    inp_h = inp_w = 64
    mean = np.array(0.588, dtype=np.float32)
    std = np.array(0.193, dtype=np.float32)
    img = cv2.resize(img, (0,0), fx=inp_w / img_w, fy=inp_h / img_h, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (inp_h, inp_w, 1))
    img = img.astype(np.float32)
    img = (img/255. - mean) / std
    img = img.transpose([2, 0, 1])
    img_tensor = torch.tensor(img).unsqueeze(0)
    saliencies = compute_saliency_maps(img_tensor, label_tensor, model)
    cv2.imshow('saliency',saliencies[0].numpy())
#########################################################################
reader = easyocr.Reader(['en'])

fontpath = "./simsun.ttc" # <== 这里是宋体路径 
font = ImageFont.truetype(fontpath, 64)

#imgsz = [640,640]
#imgsz = [1280,1280]
imgsz = [2560,2560]
ssl._create_default_https_context = ssl._create_unverified_context
device = "cuda" if torch.cuda.is_available() else "cpu"
pt = 1

#model = attempt_load("yolov5/runs/train/exp17/weights/best.pt", map_location=device)  # load FP32 model
#model = attempt_load("yolov5/runs/train/exp23/weights/best.pt", map_location=device)  # load FP32 model
#model = attempt_load("yolov5/runs/train/exp24/weights/best.pt", map_location=device)  # load FP32 model
#model = attempt_load("yolov5/runs/train/exp27/weights/best.pt", map_location=device)  # add icdar2015 and class 255=>6, 0.696
#model = attempt_load("yolov5/runs/train/exp29/weights/last.pt", map_location=device)  #add ctw
#model = attempt_load("yolov5/runs/train/exp30/weights/last.pt", map_location=device)  #add ctw
model = attempt_load("yolov5/runs/train/exp37/weights/best.pt", map_location=device)  #add ctw,rects, change class
#model = attempt_load("yolov5/runs/train/exp38/best.pt", map_location=device)  #add ctw,rects, change class



#checkpoint_crnn="CRNN_Chinese_Characters_Rec/output/OWN/crnn/2021-10-06-23-30/checkpoints/checkpoint_77_acc_1.0010.pth"

#checkpoint_crnn="output/OWN/crnn/2021-10-09-08-55/checkpoints/checkpoint_99_acc_1.0008.pth" # hidden: 512, score: 0.68
checkpoint_crnn="output/OWN/crnn/2021-10-09-12-41/checkpoints/checkpoint_99_acc_1.0009.pth" # hidden: 256, score: 0.70 crnn
#checkpoint="output/OWN/crnn/2021-10-09-22-06/checkpoints/checkpoint_33_acc_0.9968.pth" # hidden: 256, score: 0.69 Resnet
checkpoint="output/OWN/crnn/2021-10-10-09-52/checkpoints/checkpoint_98_acc_1.0009.pth" # hidden: 256, score: xxx crnn+transformer
checkpoint="output/OWN/crnn/2021-10-11-15-47/checkpoints/checkpoint_24_acc_0.9801.pth" #add many data crnn,0.73
#checkpoint_crnn="output/OWN/crnn/2021-10-12-09-16/checkpoints/checkpoint_25_acc_0.9958.pth" #add many data crnn,0.735 => 0.77 fix bug(for max_i)
#checkpoint="output/OWN/crnn/2021-10-12-21-36/checkpoints/checkpoint_15_acc_0.9716.pth" #add many data crnn

checkpoint="output/OWN/crnn/2021-10-12-21-36/checkpoints/checkpoint_28_acc_0.9781.pth" #add many data crnn,add autoaugment ?
checkpoint="output/OWN/crnn/2021-10-12-21-36/checkpoints/checkpoint_31_acc_0.9788.pth" #add many data crnn,add autoaugment 0.7945
#checkpoint_crnn="output/OWN/crnn/2021-10-14-10-11/checkpoints/checkpoint_2_acc_0.9624.pth" #add many data crnn,add autoaugment 0.7945
checkpoint_crnn="output/OWN/crnn/2021-10-14-23-36/checkpoints/checkpoint_34_acc_0.9794.pth" #crnn 0.796
checkpoint_resnet="output/OWN/crnn/2021-10-16-19-09/checkpoints/checkpoint_14_acc_0.9728.pth" #0.8
checkpoint_resnet="output/OWN/crnn/2021-10-16-19-09/checkpoints/checkpoint_30_acc_0.9805.pth"

checkpoint_resnet="output/OWN/crnn/2021-12-10-12-42/checkpoints/checkpoint_12_acc_0.9716.pth"
checkpoint_resnet="output/OWN/crnn/2021-12-11-14-17/checkpoints/checkpoint_16_acc_0.9738.pth"
checkpoint_resnet="output/OWN/crnn/2021-12-16-17-16/checkpoints/checkpoint_58_acc_0.9878.pth" #0.7803
#checkpoint_resnet="output/OWN/crnn/2021-12-21-00-37/checkpoints/checkpoint_17_acc_0.8979.pth" #0.7800
checkpoint_resnet="output/OWN/crnn/2021-12-22-00-15/checkpoints/checkpoint_82_acc_0.9216.pth"
checkpoint_resnet="output/OWN/crnn/2021-12-23-00-21/checkpoints/checkpoint_89_acc_0.9240.pth"

config, args = parse_arg(cfg="CRNN_Chinese_Characters_Rec/lib/config/OWN_config.yaml",checkpoint=checkpoint_crnn)

#char_model_crnn = crnn.get_crnn(config).to(device) #origin model
char_model_resnet = EasyOCR.EasyOcrModel(config.MODEL.IMAGE_SIZE.H, 1, config.MODEL.NUM_CLASSES + 1, config.MODEL.NUM_HIDDEN,"ResNet",True).to(device).eval()

#config.MODEL.NUM_CLASSES = 2538 #old
char_model_crnn =  EasyOCR.EasyOcrModel(config.MODEL.IMAGE_SIZE.H, 1, 10750, config.MODEL.NUM_HIDDEN,"CRNN",True).to(device).eval()

print('loading pretrained model from {0}'.format(args.checkpoint))
checkpoint_crnn = torch.load(checkpoint_crnn,map_location=device)
if 'state_dict' in checkpoint_crnn.keys():
    char_model_crnn.load_state_dict(checkpoint_crnn['state_dict'])
else:
    char_model_crnn.load_state_dict(checkpoint_crnn)

checkpoint_resnet = torch.load(checkpoint_resnet,map_location=device)
if 'state_dict' in checkpoint_resnet.keys():
    char_model_resnet.load_state_dict(checkpoint_resnet['state_dict'])
else:
    char_model_resnet.load_state_dict(checkpoint_resnet)

class EnsembleCharModel(torch.nn.Module):
    def __init__(self, *models):
        super(EnsembleCharModel, self).__init__()
        self.models = models
        self.len = len(self.models)
    def forward(self,img, text=None, is_train=True):
        result = None
        for model in self.models:
            result = model(img) if result is None else (result+model(img))
            #print('result shape',result.shape)
        return F.log_softmax(result*(1/self.len), dim=2)

char_model = EnsembleCharModel(char_model_resnet).eval()

converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

stride = int(model.stride.max())  # model stride
names = model.module.names if hasattr(model, 'module') else model.names  # get class names
imgsz = check_img_size(imgsz, s=stride)

all_path = natsorted(glob.glob("ai_data/public/*"))+natsorted(glob.glob("ai_data/private/*"))
def calc_angle(vector1,vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product)
    return angle/math.pi*180
def is_same_direction(a,b,c,d,angle_thres=20):
    vector1 = [b[0]-a[0],b[1]-a[1]]
    vector2 = [d[0]-c[0],d[1]-c[1]]
    angle = calc_angle(vector1,vector2)
    if angle_thres>=angle>=0 or 180>=angle>=(180-angle_thres):
        return True
    else:
        return False
def detect_string(path,csv_writer=None,show_target=False):
    half = False
    dataset = LoadImages(path, img_size=imgsz, stride=stride, auto=pt)
    classes=None
    conf_thres=0.2 #0.25  # confidence threshold
    iou_thres=0.5  # NMS IOU threshold
    max_det=1000 # maximum detections per image
    agnostic_nms=False,  # class-agnostic NMS

    for path, img, im0s, _,_ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]
        pred = model(img,augment=True)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        for i, det in enumerate(pred):
            p, s, im0, _ = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            print("i:",i,det)
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                frame = im0
                frame_org = frame.copy()
                x_shape, y_shape = frame.shape[1], frame.shape[0]
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                bboxes_char = []
                bboxes_string = []
                bboxes_dont_care = []
                max_conf = 0
                for *row, conf, id_dev in reversed(det):
                    id = int(id_dev.item())
                    x0, y0, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    if x0<0:
                        x0 = 0
                    if y0<0:
                        y0 = 0
                    if x2>x_shape:
                        x2 = x_shape
                    if y2>y_shape:
                        y2 = y_shape
                        
                    print('id',id)
                    if id ==1:
                        if conf>max_conf:
                            max_conf = conf
                        bboxes_char += [[(x0+x2)//2,(y0+y2)//2,abs(x2-x0),abs(y2-y0),conf]] #x,y,w,h,conf
                    elif id==0:
                        bboxes_string += [[(x0+x2)//2,(y0+y2)//2,abs(x2-x0),abs(y2-y0)]] #x,y,w,h
                    else:
                        bboxes_dont_care += [[x0,y0,x2,y2]]

                bboxes_char.sort()
                all_polygon = [[k,Polygon([[x-w//2,y-h//2],[x+w//2,y-h//2],[x+w//2,y+h//2],[x-w//2,y+h//2]])] for k,(x,y,w,h,_) in enumerate(bboxes_char)]
                for j,(x,y,w,h,_) in list(enumerate(bboxes_char))[::-1]:
                    cur_poly = Polygon([[x-w//2,y-h//2],[x+w//2,y-h//2],[x+w//2,y+h//2],[x-w//2,y+h//2]])
                    cur_area = cur_poly.area
                    for k,poly in all_polygon:
                        if j!=k and (poly.intersection(cur_poly).area/poly.union(cur_poly).area)>0.7 and cur_area>poly.area:
                            bboxes_char.pop(j)
                
                
                if len(bboxes_string)==0:
                    min_x = x_shape
                    min_y = y_shape
                    max_x = 0
                    max_y = 0
                    for x,y,w,h,conf in bboxes_char:
                        x1 = x-w//2
                        y1 = y-h//2
                        x2 = x+w//2
                        y2 = y+h//2
                        min_x = min(x1,x2,min_x)
                        max_x = max(x1,x2,max_x)
                        min_y = min(y1,y2,min_y)
                        max_y = max(y1,y2,max_y)
                    bboxes_string = [[(min_x+max_x)//2,(min_y+max_y)//2,(max_x-min_x),(max_y-min_y)]]
                
                bboxes_string.sort()
                all_polygon_str = [Polygon([[x-w//2,y-h//2],[x+w//2,y-h//2],[x+w//2,y+h//2],[x-w//2,y+h//2]]) for x,y,w,h in bboxes_string]
                for j,(x,y,w,h) in list(enumerate(bboxes_string))[::-1]:
                    cur_poly = Polygon([[x-w//2,y-h//2],[x+w//2,y-h//2],[x+w//2,y+h//2],[x-w//2,y+h//2]])
                    cur_area = cur_poly.area
                    for k,poly in list(enumerate(all_polygon_str))[::-1]:
                        intersect = poly.intersection(cur_poly).area
                        if j!=k and intersect/min(cur_area,poly.area)>0.95 and cur_area<poly.area: #remove small box in a big box
                            bx,by,bw,bh = bboxes_string[k]
                            min_x = min(x-w//2,bx-bw//2)
                            min_y = min(y-h//2,by-bh//2)
                            max_x = max(x+w//2,bx+bw//2)
                            max_y = max(y+h//2,by+bh//2)
                            bboxes_string[k] = [(min_x+max_x)//2,(min_y+max_y)//2,(max_x-min_x),(max_y-min_y)]
                            all_polygon_str[k] = Polygon([[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]])
                            bboxes_char += [[x,y,w,h,100]]
                            bboxes_string.pop(j)
                            all_polygon_str.pop(j)
                            break
                
                bgr = (0, 0, 255)

                all_polygon_str = [Polygon([[x-w//2,y-h//2],[x+w//2,y-h//2],[x+w//2,y+h//2],[x-w//2,y+h//2]]) for x,y,w,h in bboxes_string]
                b_group = [[] for bbox in  bboxes_string]
                if len(b_group)==0:
                    b_group = [[]]
                bboxes_char.sort()
                bboxes_string.sort()
                all_char_polygon_str = []

                for x,y,w,h,conf in bboxes_char:
                    if conf>max_conf-0.4:
                        x1 = x-w//2
                        y1 = y-h//2
                        x2 = x+w//2
                        y2 = y+h//2
                        cur_poly = Polygon([[x-w//2,y-h//2],[x+w//2,y-h//2],[x+w//2,y+h//2],[x-w//2,y+h//2]])
                        
                        max_i = -1
                        max_area = 0
                        for j,poly in enumerate(all_polygon_str):
                            iarea = poly.intersection(cur_poly).area
                            if max_area < iarea:
                                max_area = iarea
                                if max_area/cur_poly.area>0.4:
                                    max_i = j
                        print("max_i",max_i,len(all_polygon_str))
                        if conf==100:
                            label = ""
                        else:

                            label = recognition(config, img[max(y1,0):y2,max(x1,0):x2], char_model, converter, device)

                            if label == "#":
                                label = "###"
                            #if label !="###" and label !="":
                            #   img[max(y1,0):y2,max(x1,0):x2] = 255
                        """                        
                        label = reader.recognize(img[max(y1-10,0):y2+10,max(x1-10,0):x2+10],reformat=False)
                        if len(label)>0 and label[0][2]>0.05 and label[0][1] and len(re.sub(r'[\u4e00-\u9fff]+',"", label[0][1]))==0:
                            print(label)
                            label = label[0][1]
                        else:
                            label = recognition(config, img[max(y1-10,0):y2+10,max(x1-10,0):x2+10], char_model, converter, device)
                        """
                        if show_target:
                            #plot_map(img=img[y1:y2,x1:x2],model=char_model)
                            print(label)
                            #cv2.imshow("char",img[y1:y2,x1:x2])
                            #cv2.waitKey()
                            img_pil = Image.fromarray(frame)
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((x1, y1-64),  label, font = font, fill = (255,0,255))
                            frame = np.array(img_pil)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 3)

                        if max_i != -1:
                            bx,by,bw,bh = bboxes_string[max_i]
                            min_x = max(min(bx-bw//2,x1),0)
                            max_x = min(max(bx+bw//2,x2),x_shape)
                            min_y = max(min(by-bh//2,y1),0)
                            max_y = min(max(by+bh//2,y2),y_shape)
                            bboxes_string[max_i] = [(min_x+max_x)//2,(min_y+max_y)//2,(max_x-min_x),(max_y-min_y)]
                            b_group[max_i] += [[x,y,w,h,label]]
                            k = max_i
                            l = len(b_group[max_i]) - 1
                        else:
                            all_polygon_str += [Polygon([[x-w//2,y-h//2],[x+w//2,y-h//2],[x+w//2,y+h//2],[x-w//2,y+h//2]])]
                            bboxes_string += [[x,y,w,h]]
                            b_group.append([[x,y,w,h,label]])
                            k = len(b_group) - 1
                            l = len(b_group[-1]) - 1
                        all_char_polygon_str.append([cur_poly,label,[k,l]])
                                         
                for j,group in enumerate(b_group):
                    if len(group)==0:
                        b_group[j] = [[*bboxes_string[j],'']]

                for j,group in enumerate(b_group):
                    for k,g in enumerate(group):
                        if g[-1]=="":
                            ignore_flag = False
                            delete_index = None
                            x1 = max(g[0]-g[2]//2,0)
                            y1 = max(g[1]-g[3]//2,0)
                            x2 = g[0]+g[2]//2
                            y2 = g[1]+g[3]//2
                            if (x2-x1)/(y2-y1)>=2:
                                have_text_flag = True
                            else:
                                have_text_flag = False
                            for l,group_target in enumerate(b_group):
                                for m,(x,y,w,h,label) in enumerate(group_target):
                                    if label!="" and (not (j==l and k==m)) and g[0]-g[2]//2<x<g[0]+g[2]//2 and g[1]-g[3]//2<y<g[1]+g[3]//2:
                                        if is_english_or_num(label):
                                            delete_index = [l,m]
                                            break
                                        else:
                                            if have_text_flag:
                                                if len(label)==1:
                                                    delete_index = [l,m]
                                                else:
                                                    if abs(x-x1)<abs(x-x2):
                                                        x1 = x+w//2
                                                    else:
                                                        x2 = max(x-w//2,0)
                                            else:
                                                ignore_flag = True
                                                break

                                if ignore_flag or delete_index:
                                    break
                            if x2>x1+10 and not ignore_flag:
                                ocr_tmp = ocr.ocr(img[y1:y2,x1:x2], det=False)
                                label_tmp = ""
                                if len(ocr_tmp)>0:
                                    ocr_sub_tmp = ocr_tmp[0]
                                    if ocr_sub_tmp[1]>0.8:
                                        text_tp = ocr_sub_tmp[0]
                                        label_tmp += text_tp
                                    else:
                                        print('label not add:',ocr_sub_tmp[0])
                                if label_tmp!="":
                                    label = label_tmp
                                
                                label = re.sub(r",|\"| |\.|\(|\)|\[|\]|`|'|\/|\||\=|\{|\}|;|%|~|-|@|&|%|_|€|\*","",label)
                                if len(label)==1 and ord(label)>=58:
                                    label = ""
                                else:
                                    print('label',label)
                                if label:
                                    for n,(poly,_,(l_t,m_t)) in enumerate(all_char_polygon_str):
                                        if l_t==j and m_t==k:
                                            all_char_polygon_str[n][1] = label
                                            break
                                    b_group[j][k][-1] = label
                                    if delete_index:
                                        l,m = delete_index
                                        b_group[l][m][-1] = ""
                                        for n,(poly,_,(l_t,m_t)) in enumerate(all_char_polygon_str):
                                            if l_t==l and m_t==m:
                                                all_char_polygon_str[n][1] = ""
                                                break
                                        #assert 1==0
                for j,group in enumerate(b_group):
                    text = ""
                    x,y,w,h = bboxes_string[j]
                    x1,y1 = x-w//2,y-h//2
                    x2,y2 = x+w//2,y-h//2
                    x3,y3 = x+w//2,y+h//2
                    x4,y4 = x-w//2,y+h//2
                    if w/x_shape>0.01 and h/y_shape>0.01 and len(group)>1:
                        image_crop = img[y1:y3,x1:x3]
                        mean_color = int(cv2.mean(image_crop)[0])
                        tp1 = image_crop.copy()
                        ret,tp2 = (cv2.threshold(tp1,mean_color,255,cv2.THRESH_BINARY))
                        cnts, _ = cv2.findContours(tp2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        for k in range(len(cnts)):
                            c = cnts[k]
                            if len(c)>10:
                                (bd_x, bd_y, bd_w, bd_h) = cv2.boundingRect(c)
                                if bd_w/w>0.3 or bd_h/h>0.3 and 2.1>=bd_w/bd_h>=0.4:
                                    bd_real_x1 = bd_x+x1
                                    bd_real_y1 = bd_y+y1
                                    bd_real_x2 = bd_x+bd_w+x1
                                    bd_real_y2 = bd_y+bd_h+y1
                                    cur_poly = Polygon([[bd_real_x1,bd_real_y1],[bd_real_x2,bd_real_y1],[bd_real_x2,bd_real_y2],[bd_real_x1,bd_real_y2]])
                                    cur_poly_area = cur_poly.area
                                    have_text_flag = False
                                    for poly,label,_ in all_char_polygon_str:
                                        iarea = poly.intersection(cur_poly).area
                                        if label and iarea/cur_poly_area > 0.5 or iarea/poly.area>0.5:
                                            have_text_flag = True
                                            break
                                    if not have_text_flag:
                                        print('not have text')
                                        print(bd_x, bd_y, bd_w, bd_h)
                                        ocr_tmp = ocr.ocr(image_crop[bd_y:(bd_y+bd_h),bd_x:(bd_x+bd_w)], det=False)
                                        if show_target:
                                            print(ocr_tmp)
                                            #cv2.imshow("char",image_crop[bd_y:(bd_y+bd_h),bd_x:(bd_x+bd_w)])
                                            #cv2.waitKey()
                                        
                                        if len(ocr_tmp)>0:
                                            for ocr_sub_tmp in ocr_tmp:
                                                if ocr_sub_tmp[0] and (ocr_sub_tmp[1]>0.3 and re.sub(r'[0-9]+','',ocr_sub_tmp[0])=='') or ocr_sub_tmp[1]>0.9:
                                                    text_tp = ocr_sub_tmp[0]
                                                    all_char_polygon_str.append([cur_poly,text_tp,[j,len(b_group[j])]])
                                                    b_group[j] += [[(bd_real_x1+bd_real_x2)//2,(bd_real_y1+bd_real_y2)//2,(bd_real_x2-bd_real_x1),(bd_real_y2-bd_real_y1),text_tp]]

                print('b_group',b_group)
                new_b_group = []
                new_bboxes_string = []
                all_direct = []
                for j,group in enumerate(b_group):
                    if len(group)>1:
                        new_tmp_groups = [[] for g in group]
                        group_flags = [False for g in group]
                        group = sorted(group,key=lambda g: g[0])
                        pools = list(enumerate(copy.deepcopy(group)))
                        w_hs = []
                        text = ""
                        for x,y,w,h,label in group:
                            if label and re.sub("#","",label)!="":
                                text += label
                            if len(label)==1:
                                w_hs += [w,h]
                        if len(w_hs)==0:
                            thres = group[0][2] #min(group[0][2],group[0][3])
                        else:
                            thres = max(w_hs)
                            print('max')
                        thres = thres*0.7
                        print('thres',thres)
                        run_indexes = []
                        indexes = list(range(len(group)))
                        k =0
                        print('indexes',indexes,group)
                        while 1:
                            print('k',k)
                            run_indexes.append(k)
                            x,y,w,h,label = group[k]
                            try :
                                indexes.pop(indexes.index(k))
                                res = "Element found"
                            except ValueError :
                                res = "Element not in list !"
                            if label:
                                min_dis = None
                                min_index = -1
                                pool_pop_index = -1
                                for pop_p,(p,pool) in enumerate(pools):
                                    if p!=k and pool[-1]:
                                        dis = ((x-pool[0])**2+(y-pool[1])**2)**(1/2)
                                        if is_same_direction([0,0],[1,0],pool,[x,y]): #horizontal
                                            dis -= (pool[2]//2+w//2)
                                        else:
                                            dis -= (pool[3]//2+h//2)
                                        if (min_dis is None or min_dis>dis) and k not in new_tmp_groups[p]:
                                            x1_c,y1_c,w1,h1,label1 = group[k]
                                            x2_c,y2_c,w2,h2,label2 = pool
                                            
                                            x1_min = x1_c - w1//2
                                            x1_max = x1_c + w1//2
                                            y1_min = y1_c - h1//2
                                            y1_max = y1_c + h1//2

                                            x2_min = x2_c - w2//2
                                            x2_max = x2_c + w2//2
                                            y2_min = y2_c - h2//2
                                            y2_max = y2_c + h2//2

                                            image_crop_1 = frame_org[y1_min:y1_max,x1_min:x1_max]
                                            image_crop_2 = frame_org[y2_min:y2_max,x2_min:x2_max]

                                            mean_color_1 = np.array(cv2.mean(image_crop_1))
                                            mean_color_2 = np.array(cv2.mean(image_crop_2))

                                            color_ratio = mean_color_1/mean_color_2
                                            if (max(color_ratio)-min(color_ratio))<=0.3 and 0.5<=min(w1,h1)/min(w2,h2)<=2:
                                                min_dis = dis
                                                min_index = p
                                                pool_pop_index = pop_p
                                print(group[k],'min_dis',min_dis)
                                if min_index != -1 and min_dis<=thres:
                                    print(group[k],group[min_index])
                                    pools.pop(pool_pop_index)
                                    new_tmp_groups[k].append(min_index)
                                    new_tmp_groups[min_index].append(k)
                                    k = min_index
                                elif len(indexes)>0:
                                    k = indexes[0]

                            elif len(indexes)>0:
                                for p_index,(p,pool) in enumerate(pools):
                                    if p==k:
                                        pools.pop(p_index)
                                        break                                        
                                k = indexes[0]
                            if len(indexes)==0:
                                break
                        print(text,group,thres,new_tmp_groups)
                        new_groups = []
                        for k in run_indexes:
                            g = group[k]
                            if g[-1]:
                                match_index = -1
                                if len(new_tmp_groups[k])>0:
                                    for l,new_group in enumerate(new_groups):
                                        if len(new_tmp_groups[k])==1:
                                            if new_tmp_groups[k][0] in new_group:
                                                if len(new_groups[l])==1 or is_same_direction(group[new_groups[l][-1]],group[new_groups[l][-2]],group[k],group[new_tmp_groups[k][0]]):
                                                    match_index = l
                                                    break
                                        elif new_tmp_groups[k][0] in new_group:
                                            if len(new_groups[l])==1 or is_same_direction(group[new_groups[l][-1]],group[new_groups[l][-2]],group[k],group[new_tmp_groups[k][0]]):
                                                match_index = l
                                                break
                                        elif new_tmp_groups[k][1] in new_group:
                                            if len(new_groups[l])==1 or is_same_direction(group[new_groups[l][-1]],group[new_groups[l][-2]],group[k],group[new_tmp_groups[k][1]]):
                                                match_index = l
                                                break
                                print('text123',text,g[-1],match_index,new_groups)
                                if not group_flags[k]:
                                    group_flags[k] = True
                                    if match_index==-1 or (len(group[k][-1])>3 and is_english_or_num(group[k][-1])):
                                        new_groups.append([k])
                                    elif k not in new_groups[match_index]:
                                        sum_text = ''.join([group[l][-1] for l in new_groups[match_index] if group[l][-1]!="###"])
                                        if is_english_or_num(sum_text) and len(sum_text)>3 and not is_english_or_num(group[k][-1]):
                                            new_groups.append([k])
                                        else:
                                            new_groups[match_index].append(k)
                        for new_group in new_groups:
                            new_gs = []
                            min_x = bboxes_string[j][2]//2+bboxes_string[j][0]
                            min_y = bboxes_string[j][3]//2+bboxes_string[j][1]
                            max_x = 0
                            max_y = 0
                            direct = [0,0]
                            prev_center = None
                            new_group = sorted(new_group,key=lambda g: g)
                            for k in new_group: #new_group just include group indexes
                                g = group[k]
                                if prev_center is None:
                                    cur_direct = [0,0]
                                else:
                                    cur_direct = [g[0]-prev_center[0],g[1]-prev_center[1]]
                                    if direct!=[0,0]:
                                        if cur_direct[0]*direct[0]<0: #cur_direct in a direction opposite to direct
                                            cur_direct[0] = cur_direct[0]*(-1)
                                            cur_direct[1] = cur_direct[1]*(-1)
                                    direct[0] += cur_direct[0]
                                    direct[1] += cur_direct[1]
                                new_gs.append(g)
                                min_x = min(g[0]-g[2]//2,min_x)
                                min_y = min(g[1]-g[3]//2,min_y)
                                max_x = max(g[0]+g[2]//2,max_x)
                                max_y = max(g[1]+g[3]//2,max_y)
                                prev_center = [g[0],g[1]]
                            cx = (max_x+min_x)//2
                            cy = (max_y+min_y)//2
                            cw = (max_x-min_x)
                            ch = (max_y-min_y)
                            new_b_group.append(new_gs)
                            new_bboxes_string.append([cx,cy,cw,ch])
                            if len(new_group)==1:
                                if len(new_gs[0][-1])>1:#len(label)>1
                                    if cw>ch*0.5: #horizontal
                                        all_direct.append([1,0])
                                    else:
                                        all_direct.append([0,1])
                                else:
                                    all_direct.append(None)
                            else:
                                all_direct.append(direct)
                    elif len(group)==1:
                        new_b_group.append(group)
                        new_bboxes_string.append(group[0][0:4])
                        cw = group[0][2]
                        ch = group[0][3]
                        if cw>ch*3: #horizontal
                            all_direct.append([1,0])
                        elif ch>cw*3:
                            all_direct.append([0,1])
                        else:
                            all_direct.append(None)
                

                remove_indexes = []
                ###########################################
                
                horizontal_count = 0
                vertical_count = 0
                none_direct_count = 0
                for direct in all_direct:
                    if direct is None:
                        none_direct_count += 1
                    elif is_same_direction([0,0],[1,0],[0,0],direct):
                        horizontal_count += 1
                    else:
                        vertical_count += 1
                set_horizontal_first_flag = True if horizontal_count>vertical_count else False
                set_vertical_first_flag = True if horizontal_count<vertical_count else False
                if not set_horizontal_first_flag and not set_vertical_first_flag:
                    set_horizontal_first_flag = True
                ###########################################
                assert len(all_direct) == len(new_b_group) == len(new_bboxes_string)
                #remove false box
                for j,direct in enumerate(all_direct):
                    if direct is not None and ((not is_same_direction([0,0],[1,0],[0,0],direct) and set_horizontal_first_flag) or (not is_same_direction([0,0],[0,1],[0,0],direct) and set_vertical_first_flag)):
                        print('detect whether to destroy',new_b_group[j])
                        group = new_b_group[j]
                        if len(group)<=1:
                            continue
                        destroy_box_flag = False
                        for k,(x,y,w,h,label) in enumerate(group):
                            destroy_flag = False
                            thres = min(w,h)*0.7
                            for q,direct_target in enumerate(all_direct):
                                if q not in remove_indexes and q!=j:
                                    group_target = new_b_group[q]
                                    min_dis2 = None
                                    near_point = None
                                    pool = None
                                    for i1,g_t in enumerate(group_target):
                                        dis = ((x-g_t[0])**2+(y-g_t[1])**2)**(1/2)
                                        if is_same_direction([0,0],[1,0],[x,y],[g_t[0],g_t[1]]):
                                            dis -= (w+g_t[2])//2
                                        else:
                                            dis -= (h+g_t[3])//2
                                        if (min_dis2 is None or min_dis2>dis) and dis<thres:
                                            if (set_horizontal_first_flag and is_same_direction([0,0],[1,0],[x,y],[g_t[0],g_t[1]])) or (set_vertical_first_flag and is_same_direction([0,0],[0,1],[x,y],[g_t[0],g_t[1]])):
                                                min_dis2 = dis
                                                near_point = [g_t[0],g_t[1]]
                                                pool = g_t
                                        print('near_point',near_point)

                                    if near_point:
                                        print('g',group[k],'pool',pool)
                                        x1_c,y1_c,w1,h1,label1 = group[k]
                                        x2_c,y2_c,w2,h2,label2 = pool
                                        
                                        x1_min = x1_c - w1//2
                                        x1_max = x1_c + w1//2
                                        y1_min = y1_c - h1//2
                                        y1_max = y1_c + h1//2

                                        x2_min = x2_c - w2//2
                                        x2_max = x2_c + w2//2
                                        y2_min = y2_c - h2//2
                                        y2_max = y2_c + h2//2

                                        image_crop_1 = frame_org[y1_min:y1_max,x1_min:x1_max]
                                        image_crop_2 = frame_org[y2_min:y2_max,x2_min:x2_max]

                                        mean_color_1 = np.array(cv2.mean(image_crop_1))
                                        mean_color_2 = np.array(cv2.mean(image_crop_2))

                                        color_ratio = mean_color_1/mean_color_2
                                    
                                        if (max(color_ratio)-min(color_ratio))<=0.1 and 0.5<=min(w1,h1)/min(w2,h2)<=2:
                                            print('color_ratio',color_ratio)
                                            destroy_flag = True
                                        if destroy_flag:
                                            break
                            if not destroy_flag:
                                destroy_box_flag = False
                                print('not_destroy_flag',group[k])
                                break
                            else:
                                destroy_box_flag = True  
                        if destroy_box_flag:
                            print('destroy_flag',new_b_group[j])
                            remove_indexes.append(j)
                            for g in group:
                                new_bboxes_string.append(g[0:4])
                                new_b_group.append([g])
                                all_direct.append(None)
                                none_direct_count += 1
                    else:
                        print('--------------')
                ###########################################
                
                if none_direct_count>0:
                    for j,direct in enumerate(all_direct):
                        if direct is None and j not in remove_indexes  and not is_english_or_num(new_b_group[j][0][-1]):
                            g = new_b_group[j][0]
                            x,y,w,h,label = g
                            min_dis = x_shape**2+y_shape**2
                            min_dis_tmp = x_shape**2+y_shape**2
                            min_index = -1 
                            min_index_tmp = -1
                            for k,group in enumerate(new_b_group):
                                iter_direct = all_direct[k]
                                if j!=k and k not in remove_indexes:
                                    for l,g_t in enumerate(group):
                                        x_t,y_t,w_t,h_t,label_t = g_t
                                        if label_t and not is_english_or_num(label_t):
                                            dis = ((x_t-x)**2+(y_t-y)**2)**(1/2)
                                            mean_dis = None
                                            if all_direct[k]:
                                                prev_p = None
                                                for p in new_b_group[k]:
                                                    if prev_p is None:
                                                        prev_p = [p[0],p[1]]
                                                    else:
                                                        if mean_dis is None:
                                                            mean_dis = ((p[0]-prev_p[0])**2+(p[1]-prev_p[1])**2)**(1/2)
                                                        else:
                                                            mean_dis = (mean_dis + ((p[0]-prev_p[0])**2+(p[1]-prev_p[1])**2)**(1/2))/2
                                            cur_direct = [x_t-x,y_t-y]
                                            if mean_dis is None:
                                                thres = max(w_t,h_t)*0.3
                                                if is_same_direction([0,0],[1,0],[0,0],cur_direct):
                                                    dis -= (w_t+w)//2
                                                else:
                                                    dis -= (h_t+h)//2
                                            else:
                                                thres = mean_dis*1.2
                                            if dis>thres or (iter_direct is not None and not is_same_direction([0,0],iter_direct,[0,0],cur_direct)):
                                                continue
                                            if is_same_direction([0,0],[1,0],[0,0],cur_direct) and set_horizontal_first_flag:
                                                if min_dis>dis:
                                                    min_dis = dis
                                                if min_dis_tmp>dis:
                                                    min_dis_tmp = dis
                                                min_index = k
                                                min_index_tmp = k
                                            elif is_same_direction([0,0],[0,1],[0,0],cur_direct) and set_vertical_first_flag:
                                                if min_dis>dis:
                                                    min_dis = dis
                                                if min_dis_tmp>dis:
                                                    min_dis_tmp = dis
                                                min_index = k
                                                min_index_tmp = k
                                            else:
                                                if min_dis_tmp>dis:
                                                    min_dis_tmp = dis
                                                min_index_tmp = k
                            set_index = min_index #if min_index!=-1 else min_index_tmp
                            if set_index != -1:
                                new_b_group[set_index].append(g)
                                remove_indexes.append(j)
                                if all_direct[set_index] is None:
                                    g_t = new_b_group[set_index][0]
                                    all_direct[set_index] = [g_t[0]-g[0],g_t[1]-g[1]]
                                bx,by,bw,bh = new_bboxes_string[set_index]
                                min_x = [bx-bw//2]
                                min_y = [by-bh//2]
                                max_x = [bx+bw//2]
                                max_y = [by+bh//2]
                                for x,y,w,h,label in new_b_group[set_index]:
                                    min_x += [x-w//2]
                                    min_y += [y-h//2]
                                    max_x += [x+w//2]
                                    max_y += [y+h//2]
                                min_x = min(min_x)
                                min_y = min(min_y)
                                max_x = max(max_x)
                                max_y = max(max_y)
                                bx = (min_x+max_x)//2
                                by = (min_y+max_y)//2
                                bw = (-min_x+max_x)
                                bh = (-min_y+max_y)
                                new_bboxes_string[set_index] = [bx,by,bw,bh]
                
                #################################################

                #################################################
                """
                
                print('new_b_group',new_b_group)
                print('all_direct',all_direct)
                print('new_b_group',new_b_group)

                for j,direct in enumerate(all_direct):
                    if direct is None:
                        cur_x,cur_y,cur_w,cur_h = new_bboxes_string[j]
                        thres = cur_w*0.7
                        min_dis = (x_shape**2+y_shape**2)**(1/2) #set max distance in picture
                        min_index = -1
                        new_direct = None
                        for k,(x,y,w,h) in enumerate(new_bboxes_string):
                            if k!=j and k not in remove_indexes:
                                iter_direct = all_direct[k]
                                if iter_direct is not None and not is_same_direction([0,0],iter_direct,[cur_x,cur_y],[x,y]):
                                    continue
                                dis = ((cur_x-x)**2+(cur_y-y)**2)**(1/2)
                                print('dis_prev',dis)
                                new_direct = iter_direct if iter_direct is not None else [0,0]
                                new_direct[0] += (cur_x-x)
                                new_direct[1] += (cur_y-y)
                                if (iter_direct is None and is_same_direction([0,0],[1,0],[x,y],[cur_x,cur_y])) or (iter_direct is not None and is_same_direction([0,0],[1,0],[0,0],iter_direct)): #horizontal
                                    dis -= (w//2+cur_w//2)
                                else:
                                    dis -= (h//2+cur_h//2)
                                print('dis',dis,new_direct,new_b_group[j],'merge',new_b_group[k])
                                if min_dis>dis and dis<thres:
                                    min_dis = dis
                                    min_index = k
                        if min_index != -1:
                            print('merge_index',min_index,new_b_group[min_index],new_b_group[j])
                            #assert 1==0
                            remove_indexes.append(min_index)
                            new_b_group[j] = new_b_group[min_index]+new_b_group[j]
                            box = new_bboxes_string[min_index]
                            min_x = box[2]+cur_w
                            min_y = box[3]+cur_h
                            max_x = 0
                            max_y = 0
                            for k,(x,y,w,h,label) in enumerate(group):
                                min_x = min(x-w//2,min_x)
                                min_y = min(y-h//2,min_y)
                                max_x = max(x+w//2,max_x)
                                max_y = max(y+h//2,max_y)
                            cx = (min_x+max_x)//2
                            cy = (min_y+max_y)//2
                            cw = (-min_x+max_x)
                            ch = (-min_y+max_y)
                            new_bboxes_string[min_index] = [cx,cy,cw,ch]
                """
                """
                for j,direct in enumerate(all_direct):
                    if direct is not None:
                        group = new_b_group[j]
                        if len(group)<=1:
                            continue
                        for k,(x,y,w,h,label) in enumerate(group):
                            for q,direct_target in enumerate(all_direct):
                                if direct != direct_target and direct_target is not None and q not in remove_indexes:
                                    group_target = new_b_group[q]
                                    destroy_flag = False
                                    for i1,g_t in enumerate(group_target):
                                        min_dis = None
                                        for i2,g_t2 in enumerate(group_target):
                                            if i1 != i2:
                                                dis = ((g_t2[0]-g_t[0])**2+(g_t2[1]-g_t[1])**2)**(1/2)
                                                if min_dis is None or min_dis>dis:
                                                    min_dis = dis
                                        min_dis2 = None
                                        near_point = None
                                        for g in group:
                                            dis = ((g[0]-g_t[0])**2+(g[1]-g_t[1])**2)**(1/2)
                                            if min_dis2 is None or min_dis2>dis:
                                                min_dis2 = dis
                                                near_point = [g[0],g[1]]
                                        if min_dis is not None and min_dis2<min_dis and is_same_direction([0,0],direct_target,near_point,[g_t[0],g_t[1]]):
                                            destroy_flag = True
                                        else:
                                            destroy_flag = False
                                            break
                                    if destroy_flag:
                                        print('destroy_flag')
                                        remove_indexes.append(q)
                                        for i1,g_t in enumerate(group_target):
                                            new_bboxes_string.append(g_t[0:4])
                                            new_b_group.append([g_t])
                                            all_direct.append(None)
                                    #not finished
                """
                """
                for j,direct in enumerate(all_direct):
                    if direct is None:
                        cur_x,cur_y,cur_w,cur_h = new_bboxes_string[j]
                        thres = cur_w*0.5
                        min_dis = (x_shape**2+y_shape**2)**(1/2) #set max distance in picture
                        min_index = -1
                        remove_flag = False
                        fit_counter = 0
                        for k,(x,y,w,h) in enumerate(new_bboxes_string):
                            if k!=j and k not in remove_indexes:
                                iter_direct = all_direct[k]
                                if iter_direct is not None and not is_same_direction([0,0],iter_direct,[cur_x,cur_y],[x,y]):
                                    continue
                                mean_wh = [(char_w+char_h)/2 for char_x,char_y,char_w,char_h,char_label in new_b_group[k] if char_label!=""]
                                mean_wh = sum(mean_wh)/len(mean_wh) if len(mean_wh)>0 else None
                                dis = ((cur_x-x)**2+(cur_y-y)**2)**(1/2)
                                if (iter_direct is None and is_same_direction([0,0],[1,0],[x,y],[cur_x,cur_y])) or (iter_direct is not None and is_same_direction([0,0],[1,0],[0,0],iter_direct)): #horizontal
                                    dis -= (w//2+cur_w//2)
                                    if x-w//2<cur_x<x+w//2:
                                        continue
                                else:
                                    dis -= (h//2+cur_h//2)
                                    if y-h//2<cur_y<y+h//2:
                                        continue
                                if x-w//2<cur_x<x+w//2 and y-h//2<cur_y<y+h//2 and len([g[-1] for g in new_b_group[j] if g[-1]!=""])>0 and mean_wh is not None:
                                    #remove_flag = True
                                    #print(new_b_group[j],'merge',new_b_group[k])
                                    break
                                if dis<thres*2 and mean_wh is not None and 0.7<cur_w/mean_wh<1.3:
                                    fit_counter += 1                                    
                                    if dis<thres and min_dis>dis:
                                        min_dis = dis
                                        min_index = k
                        if remove_flag:
                            remove_indexes.append(j)
                        elif min_index != -1 and fit_counter==1:
                            remove_indexes.append(j)
                            new_b_group[min_index] = new_b_group[min_index]+new_b_group[j]
                            box = new_bboxes_string[min_index]
                            min_x = box[2]+cur_w
                            min_y = box[3]+cur_h
                            max_x = 0
                            max_y = 0
                            for k,(x,y,w,h,label) in enumerate(new_b_group[min_index]):
                                min_x = min(x-w//2,min_x)
                                min_y = min(y-h//2,min_y)
                                max_x = max(x+w//2,max_x)
                                max_y = max(y+h//2,max_y)
                            cx = (min_x+max_x)//2
                            cy = (min_y+max_y)//2
                            cw = (-min_x+max_x)
                            ch = (-min_y+max_y)
                            new_bboxes_string[min_index] = [cx,cy,cw,ch]
                            if all_direct[min_index] is None:
                                g0 = new_b_group[min_index][0]
                                g1 = new_b_group[min_index][1]
                                all_direct[min_index] = [g0[0]-g1[0],g0[1]-g1[1]]
                """
                
                for index in sorted(remove_indexes)[::-1]:
                    all_direct.pop(index)
                    new_bboxes_string.pop(index)
                    new_b_group.pop(index)
                bboxes_string = new_bboxes_string
                b_group = new_b_group
                print(b_group)
                for j,(x,y,w,h) in enumerate(bboxes_string):
                    if all_direct[j] is None or is_same_direction([0,0],[1,0],[0,0],all_direct[j]):
                        b_group[j] = sorted(b_group[j],key=lambda g: g[0])
                    else:
                        b_group[j] = sorted(b_group[j],key=lambda g: g[1])
                
                base = Path(path).stem
                have_one_line = False
                for j,group in enumerate(b_group):
                    text = ""
                    x,y,w,h = bboxes_string[j]
                    x1,y1 = x-w//2,y-h//2
                    x2,y2 = x+w//2,y-h//2
                    x3,y3 = x+w//2,y+h//2
                    x4,y4 = x-w//2,y+h//2
                    for g in group:
                        label = "###" if "#" in g[-1] else g[-1]
                        text += label

                    if text=="" or text=="###":
                        label_tmp = reader.recognize(img[y1:y3,x1:x3],reformat=False)
                        if len(label_tmp)>0 and label_tmp[0][2]>0.2 and label_tmp[0][1]:
                            print(label_tmp)
                            text_tmp = label_tmp[0][1]
                            text_tmp = re.sub(r",|\"| |\.|\(|\)|\[|\]|`|'|\/|\||\=|\{|\}|;|%|~|-|@|&|%|_|€","",text_tmp)
                            if (len(text_tmp)==1 and label_tmp[0][2]>0.95)or len(text_tmp)>1:
                                text = text_tmp
                    elif len(group)>1:
                        img_fake = np.zeros((h,w,1), dtype=np.uint8)
                        for (c_x,c_y,c_w,c_h,label) in group:
                            if label !="":
                                c_x_min = c_x - c_w//2 - x1
                                c_y_min = c_y - c_h//2 - y1
                                c_x_max = c_x + c_w//2 - x1
                                c_y_max = c_y + c_h//2 - y1
                                print((c_x_min, c_y_min), (c_x_max, c_y_max))
                                img_fake = cv2.rectangle(img_fake, (c_x_min, c_y_min), (c_x_max, c_y_max), (255,), -1)
                        cnts, _ = cv2.findContours(img_fake, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                        if len(cnts)>0:
                            coords = np.concatenate(cnts)
                            # get rotated rectangle
                            """
                            box_x,box_y,box_w,box_h = cv2.boundingRect(coords)
                            box_x_min = box_x
                            box_y_min = box_y
                            box_x_max = box_x + box_w
                            box_y_max = box_y + box_h
                            box = [[box_x_min,box_y_min],[box_x_max,box_y_min],[box_x_max,box_y_max],[box_x_min,box_y_max]]
                            """
                            rotrect = cv2.minAreaRect(coords)
                            box = np.int0(cv2.boxPoints(rotrect))
                            box = [[b[0],b[1]] for b in box]
                            failed_flag = False
                            for bxy in box:
                                bxy[0] += x1
                                bxy[1] += y1
                                if (bxy[0]<-5 or bxy[1]<-5) or (bxy[0]>x_shape+5 or bxy[1]>y_shape+5):
                                    failed_flag = True
                                else:
                                    bxy[0] = min(x_shape,max(bxy[0],0))
                                    bxy[1] = min(y_shape,max(bxy[1],0))
                            if not failed_flag:
                                (x1,y1),(x2,y2),(x3,y3),(x4,y4) = box
                                if show_target:
                                    cv2.drawContours(frame, [np.array(box,dtype=np.int32)], 0, (255,), 2)  # green
                                    
                                    cv2.imshow('real',frame)
                                    cv2.waitKey()

                    row_data = [x1,y1,x2,y2,x3,y3,x4,y4]
                    
                    if text=="" or text.replace("#","")=="":
                        text="###"
                    else:
                        have_one_line = True
                        text = text.replace("#","")
                    if text !="###" or (not have_one_line and j==(len(b_group)-1)):
                        if csv_writer != None:
                            csv_writer.writerow([base,*row_data,text])
                        print(base,*row_data,text,sep=",")
                        if False: #show_target:
                            cv2.rectangle(frame, (x1, y1), (x3, y3), bgr, 3)
                            print('text',text)
                            print(group)
                            cv2.imshow("char",img[y1:y3,x1:x3])
                            cv2.waitKey()
                """
                for (x0,y0,x2,y2) in bboxes_dont_care:
                    row_data = [x0,y0,x2,y0,x2,y2,x0,y2]
                    if csv_writer != None:
                        csv_writer.writerow([base,*row_data,"###"])
                    print(base,*row_data,"###",sep=",")
                """

                if show_target:
                    cv2.imwrite("result.jpg",cv2.resize(frame, (x_shape//2, y_shape//2), interpolation=cv2.INTER_AREA))
                    cv2.imshow("char",cv2.resize(frame, (x_shape//2, y_shape//2), interpolation=cv2.INTER_AREA))
                    cv2.waitKey()
if __name__ == "__main__":
    sub_csv = open("submission.csv","w+", encoding='UTF8')
    csv_writer = csv.writer(sub_csv)
    for path in tqdm(all_path):
        detect_string(path,csv_writer)
    sub_csv.close()
