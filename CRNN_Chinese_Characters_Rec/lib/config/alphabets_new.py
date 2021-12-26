with open("chars_list_with_ctw_and_new_ver2.txt","r") as f:
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]
alphabet = ""
for line in lines:
    alphabet += line
alphabet += "#"
print(alphabet)
