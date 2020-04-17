from annoy import AnnoyIndex
import random
import collections
import pandas as pd
import numpy as np
import json

class classification() :
    #init
    def __init__(self, dimension) :
        self.dimension = 4
        self.wine_db = pd.read_csv("/modules/Annoy/wine_db_dj.csv")
        self.ann_model=AnnoyIndex(dimension,'euclidean')
        self.ann_model.load('/modules/Annoy/test.ann')
        self.nns_num= 100
      

    def get_label(self, target_vector) : 
        nns=ann_model.get_nns_by_vector(target_vector, nns_num)
        return wine_db['label'][nns].value_counts().idxmax()
    


