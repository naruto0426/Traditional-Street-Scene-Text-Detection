## 該專案為tbrain舉辦之**[繁體中文場景文字辨識競賽－高階賽：複雜街景之文字定位與辨識 ](https://tbrain.trendmicro.com.tw/Competitions/Details/19)**比賽實作程式，並於該比賽LeaderBoard獲得第5名( 0.790860)，隊伍名稱:  TEAM_126

## 在此聲明，專案底下引用數個github專案

## yolov5: https://github.com/ultralytics/yolov5

## Single_char_image_generator: https://github.com/rachellin0105/Single_char_image_generator

## CRNN_Chinese_Characters_Rec: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec



## **以下說明操作流程**

## 1. 程式環境準備流程

```shell
####在ubuntu20.04下，安裝nvidia驅動(僅限nvidia系列顯卡使用)
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update
$ sudo apt-get install nvidia-driver-470 #安裝nvidia驅動(需視顯卡)
####安裝cuda + cudnn
# 安裝CUDA 11.4
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
$ sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget https://developer.download.nvidia.com/compute/cuda/11.4.2/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.2-470.57.02-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.2-470.57.02-1_amd64.deb
$ sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
$ sudo apt-get update
$ sudo apt-get -y install cuda
## 設定系統環境變數
$ echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
$ echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# 安裝cuDNN v8.2.4 (September 2nd, 2021), for CUDA 11.4，需到 https://developer.nvidia.com/rdp/cudnn-download 進行下載(需要自行建立nvidia帳號)
# 請下載cuDNN Runtime Library for Ubuntu20.04 x86_64 (Deb)、cuDNN Developer Library for Ubuntu20.04 x86_64 (Deb)
# https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.4/11.4_20210831/Ubuntu20_04-x64/libcudnn8_8.2.4.15-1+cuda11.4_amd64.deb 為下載連結，但是要登陸nvidia帳號
# https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.4/11.4_20210831/Ubuntu20_04-x64/libcudnn8-dev_8.2.4.15-1+cuda11.4_amd64.deb 為下載連結，但是要登陸nvidia帳號
#使用sudo dpkg -i xxx.deb安裝
$ nvdia-smi #確認cuda版本(需要預先安裝cuda + cudnn，以及nvidia驅動)
## 需先安裝python3(測試環境為python3.8)
$ pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```
## 2. 訓練/預測程式下載及相依套件安裝

```shell
# 下載 訓練及預測程式.zip( https://drive.google.com/file/d/1W_s8oX4udl5U9z_-mWBoDdkCHWnudFQI/view?usp=drive_web )
$ unzip 訓練及預測程式.zip
# chars_list.txt為所有在主辦方訓練集出現的中文字
# chars_list_with_ctw.txt為所有在主辦方訓練集出現的中文字 + CTW中的繁體字
# chars_list_with_ctw_and_new.txt為所有在主辦方訓練集出現的中文字 + CTW中的繁體字 + CharactersTrimPad28
$ pip3 install -r requirements.txt #放在壓縮黨內的主目錄

```

## 3. 下載資料集以及處理label輸入格式

```shell
# 到 https://tbrain.trendmicro.com.tw/Competitions/Details/16
# 下載train資料集(train.zip)
# 下載測試集(public.zip、private.zip)
$ unzip train.zip
$ unzip public.zip
$ unzip private.zip
## 將train資料集中的json資料檔案轉換為txt，存成jpg(方便後續訓練yolov5和文字分類)
$ python3 json2crop_npy.py
```
## 4. 訓練繁體中文檢測 (yolov5)

```shell
# 原作github網址: https://github.com/ultralytics/yolov5
$ cd yolov5
# 可自行編輯data/tw_street.yaml中的train/val路徑
$ ./train_tw.sh
# 若需驗證yolov5訓練的效果，可以val_tw.sh中，weights的路徑
$ ./val_tw.sh
```

## 5. 使用Single_char_image_generator生成訓練資料

```shell
# https://github.com/rachellin0105/Single_char_image_generator
$ cd Single_char_image_generator
$ ./gen.sh
$ cd ..
```

## 6. 預處理CTW資料集

```shell
# 下載資料集: https://ctwdataset.github.io/downloads.html
$ python3 process_ctw.py
```

## 7. 預處理CharactersTrimPad28資料集

```shell
# 至該篇文章查看詳細說明及下載連結: https://medium.com/@peterburkimsher/making-of-a-chinese-characters-dataset-92d4065cc7cc
# 預處理CharactersTrimPad28資料集，生成new_chars/dataset.txt
$ python3 new_chars.py
# 生成new_chars/dataset_train.txt (包含所有主辦方提供訓練集的中文字及CTW所出現的中文字)
$ python3 new_chars_split.py
# 生成new_chars/dataset_train_all.txt (每個繁體字只篩選其中的250個字作為訓練集)
$ python3 new_chars_split_all.py
```

## 8. 預處理ICDAR 2019-ReCTS資料集

```shell
#下載資料集: https://rrc.cvc.uab.es/?ch=12&com=introduction
#解壓縮至rects目錄，並執行以下指令
$ python3 process_rects.py
```

## 9. 訓練文字分類

```shell
# 原作的github網址: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec，我們將其修改成訓練及驗證資料集能夠分開成多個JSON_FILE和ROOT，並且finetune可以運用在已經訓練好的權重與當前訓練模型不同框架，並加入autoaugmentation，和修正其loss計算只能使用cpu的問題。
# 可至CRNN_Chinese_Characters_Rec/lib/config/OWN_config.yaml調整DATASET，每個train和val的JSON_FILE會對應到不同的ROOT。可利用TRAIN裡面的IS_FINETUNE和FREEZE來使用不同模型的weight進行finetune，finetune時能夠從FINETUNE_CHECKPOINIT繼承能夠沿用的參數。
# 最終使用的模型有參考該github做修改: https://github.com/JaidedAI/EasyOCR
$ ./train_char.sh
```

## 10. 生成預測CSV

```shell
$ python3 generate_submission_result_new8_new.py
### CSV 輸出結果在當前目錄的submission.csv
```

## 11. 分析單張圖的預測結果

```shell
#可自行修改其中的path變數，generate_submission_result_new8_new.py中設置
$ python3 get_result_new.py
```

