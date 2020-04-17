
from annoy import AnnoyIndex
import random
import pandas as pd
import numpy as np
import json

class Classifier :

    #init
    def __init__(self) :
        self.dimension = 4
        self.wine_db = pd.read_csv('./modules/Annoy/wine_db_dj.csv')
        self.ann_model=AnnoyIndex(self.dimension,'euclidean')
        self.ann_model.load('./modules/Annoy/test.ann')
        self.nns_num= 100
      

    def get_label(self, target_vector) : 
        nns=self.ann_model.get_nns_by_vector(target_vector, self.nns_num)
        return self.wine_db['label'][nns].value_counts().idxmax()
    


