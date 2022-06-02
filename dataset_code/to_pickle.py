from operator import index
import os
from matplotlib import image
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import check_array_range

def load_datas():
    path1 = "../arc-ukiyoe-faces-main/scratch/arc_images"
    # path内のデイレクトリ下にあるフォルダ名をリストで取得.
    folders = os.listdir(path1)

    #作者名
    path2 = "../arc-ukiyoe-faces-main/scratch/"
    df=pd.read_csv(path2+"arc_metadata.csv")

    # 画像の数と作者名の数を合わせる.
    new_df = check_array_range.to_same_array_renge(folders, df)


    # 画像データをリサイズし、images に格納
    images=resize(path1, folders)

    # 絵師データを painters に格納
    painters=new_df['絵師']

    return images, painters.values

def resize(path, folders):
    images=[]
    #画像を読み込み、リサイズを行う。
    for file in folders:
        img = cv2.imread(path+"/"+file)
        img = cv2.resize(img,dsize=(64,64)) #=> 約 12,448 のリストの長さ
        #リシェイプで2次元に
        img = img.reshape((img.shape[0], -1))

        images.append(img)
    
    images=np.array(images)
    #リシェイプで2次元に
    images=images.reshape((images.shape[0], -1))

    return images



def get_plickle():
    x, y=load_datas()

    df=pd.DataFrame({"images": iter(x),
                    "painters": iter(y)})

    df.to_pickle("dataset/ukiyoe_painter.pkl")
    
    
    
    

if __name__=="__main__":
    get_plickle()
