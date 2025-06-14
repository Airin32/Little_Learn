import numpy as np 
from collections import Counter
from typing import Literal 

class KNNClasification :
    def __init__ (self,k_neightborg=3) :
        self.k_ = k_neightborg
        self.x_train = None 
        self.y_train = None 
    
    def __eucludian_distance (self,x1,x2) :
        return np.sqrt(np.sum(np.power((x1-x2),2)))
    
    def fit(self,X,Y) :
        self.x_train = X 
        self.y_train = Y 
    
    def predict(self,X) :
        global_distance = list()
        for z in range(len(X)) :
            distance = list()
            for i in range(len(self.x_train)) :
                dist = self.__eucludian_distance(self.x_train[i],X[z])
                distance.append((dist,self.y_train[i]))
            distance.sort(key= lambda tup : tup[0])
            label = [label for (_,label) in distance[:self.k_]]
            class_label = Counter(label).most_common(1)[0][0]
            global_distance.append(class_label)
            distance =list()
        return np.array(global_distance)