from annoy import AnnoyIndex
import random
import collections
import pandas as pd
import numpy as np
import json

dimension = 4 


wine_db=pd.read_csv("./wine_db.csv")


ann_sample = AnnoyIndex(dimension, 'euclidean')  # Length of item vector that will be indexed

for i in range(1, 1000): #make dummy data
    v_str = wine_db['vector'][i]
    v = json.loads(v_str)
    ann_sample.add_item(i, v)

ann_sample.build(10)
ann_sample.save('test.ann')  

# ...
ann_model=AnnoyIndex(dimension,'euclidean')
ann_model.load('test.ann')
nns_num=100
target_vector = [1,1,1,1]
nns = ann_model.get_nns_by_vector(target_vector, nns_num)
check =[]
for i in range(nns_num):
    check.append(wine_db['label'][i])
printCheck = collections.Counter(check)

print(printCheck.most_common(10))
#print(nns)
#...






