import re
with open("./chars_list_with_ctw_and_new.txt","r+") as text_file:
    ts = text_file.read().split("\n")

with open("./crop_img/crop_img.txt","r") as ds_file:
    chars = [char for path,char in re.findall(r"(.*),(.*)",ds_file.read())]

for char in chars:
    if char not in ts:
        ts += [char]

with open("./chars_list_with_ctw_and_new_ver2.txt","w+") as text_file:
    text_file.write("\n".join(ts))
