import os


files_path = "/home/amokhtar/teams/continental/mot/TCIS/STSC_ANN/"
file_names = sorted(os.listdir(files_path))
train_file = open("sweden_file_names_train.txt", "w")
val_file = open("sweden_file_names_val.txt", "w")
split_ratio = 0.8



train_count = int(split_ratio*len(file_names))

for file_name in file_names[:train_count]:
    train_file.write(file_name.split(".")[0]+"\n")

for file_name in file_names[train_count:]:
    val_file.write(file_name.split(".")[0]+"\n")


train_file.close()
val_file.close()