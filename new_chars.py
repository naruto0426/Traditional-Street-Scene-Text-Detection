import os
import hanzidentifier
from pathlib import Path
from tqdm import tqdm

remove_fonts = ['Bai zhou Ancient Chinese Font–Traditional Chinese ttf','Bai zhou yin xiang ti Font-Traditional Chinesettf','Bai zhou Ying hua shu Font-Traditional Chinesettf','Beautiful art handwritten pen Font-Simplified Chinesettf','Beautiful flower art Font-Simplified Chinesettf','Beautiful flowers patterns  (ZHSRuiXian)-yolan Font-Simplified Chinesettf','Beautiful flowers Font-Simplified Chinesettf','Bo yang 7000 ti Font-Simplified Chinesettf','bnujgwttf','Bo yang CaoTi one Font-Simplified Chinesettf','Bo yang CaoTi Two Font-Simplified Chinesettf','Bo Yɑnɡ Regular script Font-Simplified Chinesettf','Bo Yang Xing Shu Font-Simplified Chinesettf','Bo Yang Xing Shu two Font-Simplified Chinesettf','BoRan Jing Handwriting Chinese Font-Simplified Chinesettf','Bowknot girl Font-Simplified Chinese0ttf','Bowknot Hat Font-Simplified Chinese0ttf','Butterflies and flowers Font-Simplified Chinesettf','Cai Yunhan Li shu calligraphy Font-Simplified Chinese ttf','Chen ji shi Ying bi Xing shu Font-Simplified Chinesettf','Childish flowers and love Font-Simplified Chinesettf','Computer symbols Font-Simplified Chinesettf','Cool Banqiao Zheng Semi-Cursive Script Font(Demo) -Traditional Chinesettf','Cool  Mark pen Handwritten Chinese Font-Simplified Chinesettf','Cool pen Handwritten Chinese Font-Simplified Chinesettf','Cool World Ink Brush (Writing Brush) Chinese Font-Simplified Chinesettf','Cool World Ming Semi-Cursive Script Chinese Font-Traditional Chinesettf','Creamy bubbles Font-Simplified Chinesettf','Cute little rabbit  (Calista) Chinese Font-Simplified Chinesettf','Da Liang reinvent¨CSimplified Chinesettf','Deformation calligraphy Chinese Font-Simplified Chinesettf','Dreaming Font-Simplified Chinesettf','Fancy Font-Simplified Chinesettf','Fat girl bowknot Chinese Font-Simplified Chinesettf','GuoFu Li handwriting Font-Simplified Chinesettf','Hua kang Tong tong Font-Traditional Chinese ttf','Standard handwritten letters Font-Simplified Chinesettf','Zhong Ji han mo Mao bi ti Font-Simplified Chinesettf','Wood graffiti Chinese Font-Simplified Chinesettf']
ch_chars = [text for text in os.listdir("new_chars/CharactersTrimPad28") if hanzidentifier.is_traditional(text)]
chars = [char for char in open("chars_list_with_ctw.txt",'r').read().split('\n') if char!='']
dataset_txt = open("new_chars/dataset.txt","w+")
i = 0
for ch_char in tqdm(ch_chars):
    for path in os.listdir(f"new_chars/CharactersTrimPad28/{ch_char}"):
        base = Path(path).stem
        if base not in remove_fonts:
            print(f"new_chars/CharactersTrimPad28/{ch_char}/{path},{ch_char}",file=dataset_txt)
            i+=1
            if ch_char not in chars:
                chars += [ch_char]
dataset_txt.close()
with open("chars_list_with_ctw_and_new.txt","w+") as f:
    for char in chars:
        print(char,file=f)
print("len of chars",len(chars))
print("len of text",i)