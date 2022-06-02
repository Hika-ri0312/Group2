from operator import index
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_img():
    path = "../arc-ukiyoe-faces-main/scratch/arc_images"

    # path内のデイレクトリ下にあるフォルダ名をリストで取得.
    folders = os.listdir(path)

    #ディレクトリ内の1/30のフォルダを取得. 確認用
    n_folders = len(folders)
    test_point = round(n_folders/30)
    

    # 画像(学習データ)とラベル(教師データ)を格納。
    
    images=[]

    #画像を読み込み、リサイズを行う。
    for i,file in enumerate(folders[:test_point]):
        img = cv2.imread(path+"/"+file)
        img = cv2.resize(img,dsize=(224,224))
        images.append(img)

    
    images=np.array(images)    

    return images, test_point

def load_painter(l):
    path = "../arc-ukiyoe-faces-main/scratch/"
    df=pd.read_csv(path+"arc_metadata.csv")

    painters=df['絵師'][:l].values
    painters=np.array(painters)
    
    return painters


def load_dataset():
    x, l=load_img()
    y=load_painter(l)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


    return x_train, x_test, y_train, y_test

load_dataset()