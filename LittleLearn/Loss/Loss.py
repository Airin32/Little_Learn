from typing import Literal
import LittleLearn as ll 
class MeanSquareError :
    def __init__ (self) :
        pass
    
    def __call__ (self,y_true,y_pred) :
        if not isinstance(y_pred,ll.GradientReflector) :
            y_pred = ll.GradientReflector(y_pred)
            return y_pred.meansquareerror(y_true)

class MeanAbsoluteError :
    def __call__(self, y_true,y_pred):
        if not isinstance(y_pred,ll.GradientReflector) :
            y_pred = ll.GradientReflector(y_pred)
        return y_pred.meanabsoluteerror(y_true)

class BinaryCrossentropy :
    def __call__ (self,y_true,y_pred) :
        if not isinstance(y_pred,ll.GradientReflector) :
            y_pred = ll.GradientReflector(y_pred)
        return y_pred.binarycrossetnropy(y_true)

class CaterigocallCrossentropy :
    def __init__(self,epsilon=1e-6) :
        self.epsilon = epsilon
    def __call__ (self,y_true,y_pred) :
        if not isinstance(y_pred,ll.GradientReflector) :
            y_pred = ll.GradientReflector(y_pred)
        return y_pred.categoricallcrossentropy(y_true,self.epsilon)

class HuberLoss :
    def __init__ (self,delta=1.0) :
        self.delta = delta 
    
    def __call__ (self,y_true,y_pred) :
        if not isinstance(y_pred,ll.GradientReflector) :
            y_pred = ll.GradientReflector(y_pred)
        return y_pred.hubber_loss(y_true,self.delta)

class SparseCategoricallCrossentropy :
    def __init__(self,epsilon=1e-6) :
        self.epsilon = epsilon
    
    def __call__ (self,y_true,y_pred) :
        if not isinstance(y_pred,ll.GradientReflector) :
            y_pred = ll.GradientReflector(y_pred)
        return y_pred.sparsecategoricallcrossentropy(y_true,self.epsilon)