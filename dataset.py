import os
import glob
import pandas as pd
import cv2

path = "arc-ukiyoe-faces-main/scratch/arc_images"
# path内のデイレクトリ下にあるフォルダ名をリストで取得。
folders = os.listdir(path)
n_folders = len(folders)
harf_point = n_folders/2
harf_folders = folders
#半分のフォルダリストを作成。実験用
del harf_folders[int(harf_point):]

print(harf_folders)
""" 
# 画像(学習データ)とラベル(教師データ)を格納。
images=[]

#画像を読み込み、リサイズを行う。
for file in harf_folders:
    img = cv2.imread(path+"/"+file)
    #img = cv2.resize(img,dsize=(224,224))
    images.append(img) 

#img = cv2.imread(path+"/"+folders[0])
#images.append(img)

df = pd.DataFrame(columns=["images", "painters"])
df["images"]=images
print(df.head()) """