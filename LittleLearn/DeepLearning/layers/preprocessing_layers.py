import LittleLearn as ll 
from typing import Literal
import numpy as np 
import traceback 

class LayerNormalization :
    def __init__ (self,epsilon=1e-6,use_gradient_reflect : Literal[True,False] = True) :
        self.beta = None 
        self.gamma = None 
        self.epsilon = epsilon 
        self.use_engine_rf = use_gradient_reflect
        self.__cache = None 
    
    def __build_component (self,features) :
        self.beta = np.zeros((1,features))
        self.gamma = np.ones((1,features))
        if self.use_engine_rf :
            self.beta = ll.GradientReflector(self.beta)
            self.gamma = ll.GradientReflector(self.gamma)
    
    def __call__(self,x) :
        if self.beta is None or self.gamma is None :
            self.__build_component(x.shape[-1])
        if not self.use_engine_rf : 
            mean = np.mean(x,axis=-1,keepdims=True)
            var = np.mean(np.power((x - mean),2),axis=-1,keepdims=True)
            normal_vals = ((x - mean) / np.sqrt(var + self.epsilon))
            out = self.gamma * normal_vals+ self.beta
            self.__cache = [mean,var,normal_vals,x]
            return out 
        else :
            if not isinstance(x,ll.GradientReflector) :
                x = ll.GradientReflector(x)
            return x.layernormalization_backend(gamma=self.gamma,beta=self.beta,epsilon=self.epsilon)
    
    def backwardpass (self,Grad_out) : 
        try : 
            if self.use_engine_rf :
                raise RuntimeError("the backpropogation layers its run on Gradient reflector")
            mean,var,normal,x = self.__cache
            d_beta = np.sum(Grad_out,axis=0)
            d_gamma = np.sum((Grad_out * normal),axis=0)
            d_n = Grad_out * self.gamma 
            d_var = np.sum(d_n * (x - mean) * -0.5 * np.power(var + self.epsilon,-1.5))
            d_mean = np.sum(d_n * -1 / np.sqrt(var + self.epsilon))\
            + d_var * np.mean(-2 * (x - mean),axis=-1,keepdims=True)
            grad_out = d_n / np.sqrt(var + self.epsilon) + d_var * 2 * (x - mean)\
            / x.shape[-1] + d_mean / x.shape[-1]
            return {
                'grad_z' : grad_out,
                'd_beta' : d_beta,
                'd_gamma' : d_gamma

            }
            
        except Exception as e :
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

class BacthNormalization :
    def __init__(self,epsilon=1e-6,use_gradient_reflector : Literal[True,False] = True):
        self.epsilon = epsilon
        self.gamma = None 
        self.beta = None 
        self.__cache = None 
        self.use_gradEngine = use_gradient_reflector
    
    def __build_component (self,features) :
        if self.use_gradEngine :
            self.gamma = ll.GradientReflector(np.ones((1,features)),_op='gamma')
            self.beta = ll.GradientReflector(np.zeros((1,features)),_op='beta')
        else :
            self.gamma = np.ones((1,features))
            self.beta = np.zeros((1,features))
    
    def __call__ (self,x) :
        if self.gamma is None or self.beta is None  :
            self.__build_component(x.shape[-1])
        if not self.use_gradEngine :
            mean = np.mean(x,axis=0,keepdims=True)
            var = np.mean(np.power((x - mean),2),axis=0,keepdims=True)
            normal = (x - mean) / np.sqrt(var + self.epsilon)
            outputs = self.gamma *  normal + self.beta 
            self.__cache = (mean,var,normal,x)
            return outputs
        if not isinstance(x,ll.GradientReflector) :
            x = ll.GradientReflector(x)
        return x.bacthnormalization_backend(gamma=self.gamma,beta=self.beta,epsilon=self.epsilon)
    
    def backwardpass (self,grad_out) :
        try :
            if self.use_gradEngine is True :
                raise RuntimeError("backpropogation layers work at Gradient Reflector")
            mean,var,normal,x = self.__cache
            d_beta = np.sum(grad_out,axis=0)
            d_gamma = np.sum((grad_out * normal),axis=0)
            d_n = grad_out * self.gamma
            d_var = np.sum(d_n * (x - mean) * -0.5 * np.power((var  + self.epsilon),-1.5),axis=0,keepdims=True)
            d_mean = np.sum(d_n * -1 / np.sqrt(var + self.epsilon),axis=0,keepdims=True)\
            + d_var * np.mean((-2 * (x - mean)),axis=0,keepdims=True)
            grad = d_n / np.sqrt(var + self.epsilon) + d_var * 2 * (x - mean)\
            / x.shape[0] + (d_mean / x.shape[0])
            return {
                "grad_z" : grad,
                "grad_beta" : d_beta,
                "grad_gamma" : d_gamma
            }
        except Exception as e :
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
            
