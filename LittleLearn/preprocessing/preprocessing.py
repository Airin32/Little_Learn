import numpy as np 
import traceback

def PositionalEncoding (maxpos,d_model) :
    try:
        if maxpos == 0 or maxpos is None :
            raise ValueError(f"maxpos == {maxpos} ")
        elif d_model == 0 or d_model is None :
            raise ValueError(f"d_model == {d_model}")
        positional = np.arange(maxpos,dtype=np.float32) [:,np.newaxis]
        dimention = np.arange(d_model,dtype=np.float32)
        div_values = np.power(10000.0,(2 * (dimention//2) / np.sqrt(d_model)))
        angle_rads = positional / div_values
        angle_rads[:,0::2] = np.sin(angle_rads[:,0::2])
        angle_rads[:,1::2] = np.cos(angle_rads[:,1::2])
        return angle_rads
    except Exception as e :
        e.add_note("maxpos variable must initialization first == (PositonalEncoding(maxpos=your initialization values))")
        e.add_note("d_models variable must initialization firt == (PositionalEncoding(d_model=your dimention model values))")
        traceback.print_exception(type(e),e,e.__traceback__)
        raise 

class MinMaxScaller :
    def __init__ (self,f_range=None,epsilon=1e-6) :
        if f_range is not None :
            try : 
                if len(f_range) !=2 :
                    raise ValueError("Error : f_range must give 2 values at list [min_range,max_range] or (min_range,max_range)")
                self.r_min,self.r_max = f_range
            except Exception as e :
                traceback.print_exception(type(e),e,e.__traceback__)
        self.__range = f_range
        self.epsilon = epsilon 
        self.min = None 
        self.max = None 
    
    def fit(self,x) :
        self.min = np.min(x,axis=0)
        self.max = np.max(x,axis=0)
    
    def scaling (self,x) :
        try:
            if self.min is None or self.max is None :
                raise RuntimeError("You must fit scalers First")
            scaled = (x - self.min) / (self.max - self.min + self.epsilon)
            if self.__range is None :
                return scaled
            return scaled * (self.r_max - self.r_min) + self.r_min
        except Exception as e :
            e.add_note("do MinMaxScaller().fit(x) before do scaling")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def fit_scaling (self,x) :
        self.fit(x) 
        scaled = (x - self.min) / (self.max - self.min + self.epsilon)
        if self.__range is None :
            return scaled
        return scaled * (self.r_max - self.r_min) + self.r_min

    def inverse_scaling (self,x) :
        try :
            if np.max(self.min) == np.min(x) or np.max(self.max) == np.max(x)\
            or np.min(x) > np.max(self.max) :
                warning = RuntimeWarning("Warning :  The Values its to large for inverse")
                print(warning)
            if self.min is None or self.max is None :
                raise RuntimeError("Error : You must fit scaller first")
            if self.__range is None :
                return x * (self.max - self.min) + self.min
            unscale = (x - self.r_min) / (self.r_max - self.r_min + self.epsilon)
            unscale = unscale * (self.max - self.min) + self.min
            return unscale
        except Exception as e :
            e.add_note("You must do MinMaxScaller().fit() first")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

class StandardScaller :
    def __init__ (self,epsilon=1e-6) :
        self.epsilon = epsilon
        self.std = None 
        self.mean = None 
    
    def fit(self,x) :
        self.mean = np.mean(x,axis=0,dtype=np.float32) 
        variance = np.mean((x - self.mean)**2,axis=0,dtype=np.float32)
        self.std = np.sqrt(variance)
    
    def scaling(self,x) :
        try : 
            if self.mean is None or self.std is None :
                raise RuntimeError("you must fit scaller first")
            return (x - self.mean) / (self.std + self.epsilon)
        except Exception as e :
            e.add_note("do StandardScaller().fit() before do scalling")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

    
    def inverse_scaling(self,x) :
        try :
            if self.std is None or self.mean is None :
                raise RuntimeError("you must fit scaller first")
            return x * self.std + self.mean
        except Exception as e :
            e.add_note("do StandardScaller().fit() before do scalling")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

    def fit_scaling (self,x) :
        self.fit(x) 
        return (x - self.mean) / (self.std + self.epsilon)

class MaxAbsoluteScaller :
    def __init__ (self,epsilon = 1e-6) :
        self.epsilon = epsilon
        self.max_abs = None 
    
    def fit(self,x) :
        abs_values = np.abs(x) + self.epsilon
        self.max_abs = np.max(abs_values,axis=0)
    
    def scaling (self,x) :
        try :
            if self.max_abs is None :
                raise RuntimeError("you must fit scaller first")
            return x / self.max_abs
        except Exception as e :
            e.add_note("do MaxAbsoluteScaller().fit() before do scalling")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def fit_scalling (self,x) :
        self.fit(x) 
        return x / self.max_abs
    
    def inverse_scalling (self,x) :
        try :
            if self.max_abs is None :
                raise RuntimeError("you must fit scaller first")
            return x * self.max_abs
        except Exception as e :
            e.add_note("do MaxAbsoluteScaller().fit() before do scalling")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

class Tokenizer :
    def __init__(self):
        self.token = dict()
        self.counter = 1
    
    def fit_from_text(self,x) :
        for sentece in x :
            sentece= sentece.lower()
            for token,word in enumerate(sentece.split(),start=self.counter):
                if word not in self.token.values():
                    self.token.update({token:word})
                    self.counter +=1
    
    def word_to_sequence (self,x) :
        result = list()
        for sentence in x :
            token = []
            words = sentence.lower().split()
            for t,word in self.token.items() :
                if word in words :
                    token.append(t)
            result.append(token)
        return result
    
    def sequence_to_word (self,sequence) :
        result = list() 
        for seq in sequence :
            word = []
            for tok,w in self.token.items() :
                if seq in tok :
                    word.append(w)
            result.append(word)
        return result

