from operator import le
import os
import pandas as pd


path = "arc-ukiyoe-faces-main/scratch/arc_images"
# path内のデイレクトリ下にあるフォルダ名をリストで取得。
folders = os.listdir(path)

print(len(folders))