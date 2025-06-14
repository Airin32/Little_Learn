import numpy as np 
from typing import Literal

class Adam :
    def __init__ (self,learning_rate=0.001,Beta1=0.9,Beta2=0.999,epsilon=1e-5) :
        self.Beta1 =Beta1
        self.Beta2 = Beta2
        self.Momentum_w = None 
        self.Momentum_b = None 
        self.Rmsprop_w = None 
        self.Rmsprop_b = None 
        self.learning_rate = learning_rate
        self.epsilon = epsilon 
        self.name = 'adam'
    
    def build_component (self,features,bias) :
        self.Momentum_w = np.zeros_like(features)
        self.Momentum_b = np.zeros_like(bias)
        self.Rmsprop_w = np.zeros_like(features)
        self.Rmsprop_b = np.zeros_like(bias)
    
    def __call__ (self,weight,bias,gradient_w,gradient_b,iteration) :
        if self.Momentum_w is None or self.Rmsprop_w is None :
            self.build_component(weight,bias)
        self.Momentum_w = self.Beta1 * self.Momentum_w + (1 - self.Beta1) * gradient_w
        self.Momentum_b = self.Beta1 * self.Momentum_b + (1 - self.Beta1) * gradient_b
        self.Rmsprop_w = self.Beta2 * self.Rmsprop_w + (1 - self.Beta2) * np.power(gradient_w,2)
        self.Rmsprop_b = self.Beta2 * self.Rmsprop_b + (1 - self.Beta2) * np.power(gradient_b,2)
        Momentum_w = self.Momentum_w / (1 - np.power(self.Beta1,(iteration + 1)))
        Momentum_b = self.Momentum_b / (1 - np.power(self.Beta1,(iteration + 1)))
        Rmsprop_w = self.Rmsprop_w / (1 - np.power(self.Beta2,(iteration + 1)))
        Rmsprop_b = self.Rmsprop_b / (1 - np.power(self.Beta2,(iteration + 1)))
        weight -= self.learning_rate / (np.sqrt(Rmsprop_w + self.epsilon)) * Momentum_w
        bias -= self.learning_rate / (np.sqrt(Rmsprop_b + self.epsilon)) * Momentum_b
        return {
            'weight':weight,
            'bias':bias
        }
    
    def change_component (self,Momentum_w,Momentum_b,rms_w,rms_b) :
        self.Momentum_w = Momentum_w
        self.Momentum_b = Momentum_b
        self.Rmsprop_w = rms_w
        self.Rmsprop_b = rms_b

class Rmsprop :
    def __init__(self,learning_rate=0.001,Beta=0.999,epsilon=1e-5):
        self.Beta = Beta 
        self.rmsprop_w = None 
        self.rmsprop_b = None 
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.name='rmspop'
    
    def build_component(self,features) :
        self.rmsprop_w = np.zeros((features,1))
        self.rmsprop_b = 0.0
    
    def __call__ (self,weight,bias,gradient_w,gradient_b,iteration) :
        if self.rmsprop_w is None or self.rmsprop_b is None :
            self.build_component(weight.shape[1])
        self.rmsprop_w = self.Beta * self.rmsprop_w + (1 - self.Beta) * np.power(gradient_w,2)
        self.rmsprop_w = self.Beta * self.rmsprop_b + (1 - self.Beta) * np.power(gradient_b,2)
        rmsprop_w = self.rmsprop_w / (1 - np.power(self.Beta,(iteration + 1)))
        rmsprop_b = self.rmsprop_b / (1 - np.power(self.Beta,(iteration + 1)))
        weight -= self.learning_rate / (np.sqrt(rmsprop_w + self.epsilon)) * np.power(gradient_w,2)
        bias -= self.learning_rate / (np.sqrt(rmsprop_b + self.epsilon)) * np.power(gradient_b,2)
        return {
            'weight' : weight,
            'bias' : bias
        }
    
    def change_component (self,Rmsprop_w,Rmsrpop_b) :
        self.rmsprop_w = Rmsprop_w
        self.rmsprop_b = Rmsrpop_b


class Momentum :
    def __init__ (self,learning_rate=0.001,Beta=0.9,Task : Literal['regression','classification']='classification',epsilon=1e-5) :
        self.Beta = Beta 
        self.Task = Task 
        self.Momentum_w = None 
        self.Momentum_b = None 
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.name='momentum'
    
    def build_component (self,features) : 
        self.Momentum_w = np.zeros((features,1))
        self.Momentum_b = 0.0 
    
    def __call__ (self,weight,bias,gradient_w,gradient_b,iteration) :
        if self.Momentum_w is None or self.Momentum_b is None :
            self.build_component(weight.shape[1])
        self.Momentum_w = self.Beta * self.Momentum_w + (1 - self.Beta) * gradient_w
        self.Momentum_b = self.Beta * self.Momentum_b + (1 - self.Beta) * gradient_b
        Momentum_w = self.Momentum_w / (1 - np.power(self.Beta,(iteration + 1)))
        Momentum_b = self.Momentum_b / (1 - np.power(self.Beta,(iteration + 1)))
        weight -= self.learning_rate * Momentum_w
        bias -= self.learning_rate * Momentum_b
        return {
            'weight' : weight,
            'bias' : bias
        }

class GradientDescent :
    def __init__ (self,learning_rate=0.001) :
        self.learning_rate = learning_rate
        self.name ='gradient_descent'
    
    def __call__ (self,weight,bias,gradient_w,gradien_b) :
        weight -= self.learning_rate * gradient_w
        bias -= self.learning_rate * gradien_b
        return {
            'weight' : weight,
            'bias' : bias
        }

class AdamW :
    def __init__ (self,learning_rate,L2=1e-3,epsilon=1e-5,Beta1=0.9,Beta2=0.999) :
        self.learning_rate = learning_rate 
        self.L2 = L2
        self.epsilon = epsilon 
        self.Momentum_w = None 
        self.Momentum_b = None 
        self.Rmsprop_w = None 
        self.Rmsprop_b = None
        self.Beta1 = 0.9 
        self.Beta2 = 0.999 
        self.name = 'adamw'
    def build_component(self,features) :
        self.Momentum_w = np.zeros((features,1))
        self.Momentum_b = 0.0 
        self.Rmsprop_w = np.zeros((features,1))
        self.Rmsprop_b = 0.0
    
    def __call__ (self,weight,bias,gradient_w,gradient_b,iteration) :
        if self.Momentum_w is None or self.Rmsprop_w is None :
            self.build_component(weight.shape[1])

        self.Momentum_w = self.Beta1 * self.Momentum_w + (1 - self.Beta1) * gradient_w
        self.Momentum_b = self.Beta1 * self.Momentum_b + (1 - self.Beta1) * gradient_b
        self.Rmsprop_w = self.Beta2 * self.Rmsprop_w + (1 - self.Beta2) * np.power(gradient_w,2)
        self.Rmsprop_b = self.Beta2 * self.Rmsprop_b + (1 - self.Beta2) * np.power(gradient_b,2)
        Momentum_w = self.Momentum_w / (1 - np.power(self.Beta1,(iteration + 1)))
        Momentum_b = self.Momentum_b / (1 - np.power(self.Beta1,(iteration + 1)))
        Rmsprop_w = self.Rmsprop_w / (1 - np.power(self.Beta2,(iteration + 1)))
        Rmsprop_b = self.Rmsprop_b / (1 - np.power(self.Beta2,(iteration + 1)))
        weight_decay = 1 - self.learning_rate * self.L2
        weight -= weight_decay * (Momentum_w / np.sqrt(Rmsprop_w + self.epsilon))
        bias -= self.learning_rate / np.sqrt(Rmsprop_b + self.epsilon) * Momentum_b
        return {
            'weight' : weight,
            'bias' : bias
        }
    def change_component (self,Momentum_w,Momentum_b,rms_w,rms_b) :
        self.Momentum_w = Momentum_w
        self.Momentum_b = Momentum_b
        self.Rmsprop_w = rms_w
        self.Rmsprop_b = rms_b

class Adamax :
    def __init__ (self,learning_rate = 1e-2,epsilon=1e-5,Beta1=0.9,Beta2=0.999) :
        self.learning_rate = learning_rate 
        self.epsilon = epsilon 
        self.Beta1 = Beta1
        self.Beta2 = Beta2 
        self.Momentum_w = None 
        self.Momentum_b = None 
        self.abs_grad_w = None 
        self.abs_grad_b = None 
        self.name='adamax'
    
    def build_component (self,features) :
        self.Momentum_w = np.zeros((features,1))
        self.Momentum_b = 0.0 
        self.Rmsprop_w = np.zeros((features,1))
        self.Rmsprop_b = 0.0 
    def __call__ (self,weight,bias,gradient_w,gradient_b,iteration) :
        if self.Momentum_w is None or self.abs_grad_w is None :
            self.build_component(weight.shape[1])
        self.Momentum_w = self.Beta1 * self.Momentum_w + (1 - self.Beta1) * gradient_w
        self.Momentum_b = self.Beta1 * self.Momentum_b + (1 - self.Beta1) * gradient_b
        self.abs_grad_w = np.maximum((self.Beta2 * self.abs_grad_w),np.absolute(gradient_w))
        self.abs_grad_b = np.maximum((self.Beta2 * self.abs_grad_b),np.absolute(gradient_b))
        Momentum_w = self.Momentum_w / (1 - np.power(self.Beta1,(iteration + 1)))
        Momentum_b = self.Momentum_b / (1 - np.power(self.Beta1,(iteration + 1)))
        abs_grad_w = self.abs_grad_w / (1 - np.power(self.Beta2,(iteration + 1)))
        abs_grad_b = self.abs_grad_b / (1 - np.power(self.Beta2,(iteration + 1)))
        weight -= (self.learning_rate / abs_grad_w) * Momentum_w
        bias -= (self.learning_rate / abs_grad_b) * Momentum_b
        return {
            'weight' : weight,
            'bias' : bias
        } 
    
    def change_component (self,Momentum_w,Momentum_b,rms_w,rms_b) :
        self.Momentum_w = Momentum_w
        self.Momentum_b = Momentum_b
        self.Rmsprop_w = rms_w
        self.Rmsprop_b = rms_b