import numpy as np 
from typing import Literal 
import traceback
from LittleLearn import activations
import LittleLearn as ll 

class Dense :
    def __init__(self,units,activation : Literal['relu','softmax','sigmoid','linear'] = None,
                 initial_Weight : Literal ['uniform','normal'] = 'normal' ,
                 inital_bias : Literal ['zeros','random'] = 'zeros',Name="Dense",
                 use_Gradient_reflector:Literal[True,False] = True):
        self.units = units 
        self.activation = activation
        self.weight = None 
        self.bias = None 
        self.initial_weight = initial_Weight
        self.initial_bias = inital_bias
        self.__activation_name = None 
        self.parameter = None 
        self.Name = Name 
        self.output = None 
        self.logits = None 
        self.__Feature_flag = None 
        self.y_label = None
        self.__input = None 
        self.__ext_output = None 
        self.use_grad_ref = use_Gradient_reflector

    def __softmax (self,x) :
        if not self.use_grad_ref:
            x_max = np.max(x,axis=-1,keepdims=True)
            x_exp = np.exp(x - x_max,dtype=np.float32)
            x_sum = np.sum(x_exp,axis=-1,dtype=np.float32,keepdims=True)
            x_sum[x_sum==0] = 1e-7
            self.output= x_exp / x_sum
            return self.output
        return activations.softmax(x,self.y_label)
        
    def __sigmoid (self,x) :
        if not self.use_grad_ref:
            self.output = 1 / (1 + np.exp(-x,dtype=np.float32))
            return self.output
        return activations.sigmoid(x)
        
    def __linear (self,x) :
        if not self.use_grad_ref :
            self.__output = x 
            return self.output
        return activations.linear(x)
        
    def __relu (self,x) :
        if not self.use_grad_ref:
            return   np.maximum(0,x,dtype=np.float32)
        return activations.relu(x)
    
    def __d_relu (self,x) :
        return np.where(x > 0,1,0)
    
    def __d_sigmoid (self,x) :
        return x * (1 - x)
    
    def __d_softmax (self,y) :
        r_d = list()
        for i in range(len(y)) :
            r_d.append(self.output[i] - y[i])
        return np.vstack(r_d)
    
    def __build_weight (self,Features) :
        self.__Feature_flag = Features
        try :
            self.__activation_name = self.activation.name 
        except :
            self.__activation_name = None 
        try :
            if self.units <= 0 or self.units is None :
                raise RuntimeError("0 / None is disagreed")
            if self.initial_weight not in ['uniform','normal'] :
                raise RuntimeError("this layers just support for (normal) and uniform")
            if self.initial_bias not in ['zeros','random'] :
                RuntimeError("this layers just suport inital bias for zeros and random")
            if self.__activation_name is None :
                if self.activation in ['softmax','sigmoid'] :
                    if self.initial_weight == 'normal':
                        normal_xavier = np.sqrt(2 / (Features + self.units))
                        self.weight = np.random.normal(loc=0,scale=normal_xavier,size = (Features,self.units))
                    elif self.initial_weight == 'uniform' :
                        uniform_xavier = np.sqrt(6 / (Features + self.units))
                        self.weight = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(Features,self.units))
                else :
                    if self.initial_weight == 'normal' : 
                        Normal_He = np.sqrt(2 / Features)
                        self.weight = np.random.normal(loc=0,scale=Normal_He,size=(Features,self.units))
            
                    elif self.initial_weight == 'uniform':
                        uniform_He = np.sqrt(6 / Features) 
                        self.weight = np.random.uniform(low=-uniform_He,high=uniform_He,size=(Features,self.units))

            else :
                if self.__activation_name in ['softmax','sigmoid'] :
                    if self.initial_weight == 'normal':
                        normal_xavier = np.sqrt(2 / (Features + self.units))
                        self.weight = np.random.normal(loc=0,scale=normal_xavier,size = (Features,self.units))
                    elif self.initial_weight == 'uniform':
                        uniform_xavier = np.sqrt(6 / (Features + self.units))
                        self.weight = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(Features,self.units))

                else :
                    if self.initial_weight == 'normal' :
                        Normal_He = np.sqrt(2 / Features)
                        self.weight = np.random.normal(loc=0,scale=Normal_He,size=(Features,self.units))

                    elif self.initial_weight == 'uniform' and self.initial_bias == 'random' :
                        uniform_He = np.sqrt(6 / Features) 
                        self.weight = np.random.uniform(low=-uniform_He,high=uniform_He,size=(Features,self.units))
            
            if self.initial_bias == 'zeros' :
                self.bias = np.zeros((1,self.units))
            elif self.initial_bias == 'random' and self.initial_weight =='normal':
                Normal_He = np.sqrt(2/Features)
                self.bias = np.random.normal(loc=0,scale=Normal_He,size=(1,self.units))
            elif self.initial_bias == 'random' and self.initial_weight=='uniform' :
                uniform_He = np.sqrt(6/Features)
                self.bias = np.random.uniform(low=-uniform_He,high=uniform_He,size=(1,self.units))
            self.parameter = (Features * self.units) + self.units

        except Exception as e :
            if self.units <= 0 or self.units is None :
                e.add_note("recomended for 2 exponential values like = > example = 2,4,8,16,32....N for initial units")
            if self.initial_weight not in ['uniform','normal'] :
                e.add_note("use the available initial methode [normal,uniform]")
            if self.initial_bias not in ['zeros','random'] :
                e.add_note ("use the available initial methode [zeros,random]")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

    def __call__ (self,x) :
        if self.weight is None or self.bias is None :
            self.__build_weight(x.shape[-1])
        if x.shape[-1] != self.__Feature_flag :
            self.__build_weight(x.shape[-1])
        self.__input = x 
        if not self.use_grad_ref:
            logits = np.dot(x,self.weight) + self.bias
        else : 
            if not isinstance(self.weight,ll.GradientReflector):
                self.weight = ll.convert_to_tensor(self.weight)
            if not isinstance(self.bias,ll.GradientReflector):
                self.bias = ll.convert_to_tensor(self.bias)
            logits = ll.matmul(x,self.weight) + self.bias
        self.logits = logits
        if self.activation is not None :
            try : 
                if self.activation == 'softmax' :
                    return self.__softmax(logits)
                elif self.activation == 'sigmoid' :
                    return self.__sigmoid(logits)
                elif self.activation == 'relu' :
                    return self.__relu(logits)
                elif self.activation == 'linear' :
                    return self.__linear(logits)
            except :
                return self.activation(logits)
        self.__ext_output = logits
        return logits
        
    
    def BackwardPass (self,gradient_values) :
        try :
            if self.use_grad_ref :
                raise RuntimeError("Error : if you use Gradient Reflector all Backpropogation automatic run")
            if self.__activation_name is None  :
                if self.activation == 'softmax' :
                    d_logits = self.__d_softmax(self.y_label)
                if self.activation == 'sigmoid' :
                    d_logits = self.__d_sigmoid(self.output)
                if self.activation == 'relu' :
                    d_logits= self.__d_relu(self.logits)
                if self.activation == 'linear' :
                    d_logits = self.logits
                if self.activation is None :
                    d_logits = self.logits
                grad_next = gradient_values * d_logits
                grad_w = np.dot(self.__input.T,grad_next)
                grad_b = np.sum(grad_next,axis=0)
                grad_next = np.dot(grad_next,self.weight.T)
            return {
                'grad_z' : grad_next,
                'grad_w' : grad_w,
                'grad_b' : grad_b
            }
        except Exception as e :
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 



class Attention :
    def __init__ (self,units,Masking = False,return_scores_attention = False,
                initial_weight : Literal ['normal','uniform'] = 'normal',
                initial_bias : Literal['zeros','random'] = 'zeros',
                derivative_func : Literal['jacobian','alternative'] = 'alternative' ) :
        self.Masking = Masking 
        self.units = units 
        self.Weight_Q = None 
        self.Weight_K = None 
        self.Weight_V = None 
        self.bias_q = None 
        self.bias_k = None 
        self.bias_v = None 
        self.__cache = None 
        self.return_attn_state = return_scores_attention
        self.parameters = None 
        self.initial_weight = initial_weight
        self.initial_bias = initial_bias
        self.__input = None 
        self.__scores = None  
        self.__mask_vals = None
        self.__key_dims = None  
        self.derivative_mode = derivative_func
    
    def __alternative_Derivative_softmax(self,softmax_out,grad_scores) :
        step1 = np.sum(grad_scores * softmax_out,axis=-1,keepdims=True)
        step2 = softmax_out * (grad_scores - step1)
        return step2
        
    
    def __create_masking (self,size) :
        masking = 1 -  np.tril(np.ones(shape=(size,size),dtype=np.float32))
        return masking 
    
    def __softmax (self,x) :
        x_max = np.max(x,axis=-1,keepdims=True)
        x_exp = np.exp(x - x_max)
        x_sum = np.sum(x_exp,axis=-1,keepdims=True)
        x_sum[x_sum==0] = 1e-9
        return x_exp / x_sum
    
    def __derivative_softmax (self,x) :
        batch,seq,dim = x.shape
        grad = np.zeros((batch,seq,dim,dim),dtype=np.float32)
        for b in range(batch) :
            for s in range(seq) :
                y = x[b,s].reshape(-1,1)
                jacobian = np.diagflat(y) - np.dot(y,y.T)
                grad[b,s] = jacobian
        return grad

    def __build_weight (self,features) : 
        try :
            if self.units <= 0 or self.units is None :
                raise RuntimeError("0 / None is disagreed")
            if self.initial_weight not in ['uniform','normal'] :
                raise RuntimeError("Initial methode not available")
            if self.initial_bias not in ['random','zeros'] :
                raise RuntimeError("Initial methode not available")
            if self.initial_weight == 'normal' :
                scales_variance =  np.sqrt(2 / (features + self.units))
                self.Weight_Q = np.random.normal(loc =0 ,scale=scales_variance,size=(features,self.units)) 
                self.Weight_K = np.random.normal(loc=0,scale=scales_variance,size=(features,self.units)) 
                self.Weight_V = np.random.normal(loc=0,scale=scales_variance,size=(features,self.units)) 
            elif self.initial_weight == 'uniform' :
                scales_variance = np.sqrt(6/(features + self.units))
                self.Weight_Q = np.random.uniform(low=-scales_variance,high=scales_variance,size=(features,self.units))
                self.Weight_K = np.random.uniform(low=-scales_variance,high=scales_variance,size=(features,self.units))
                self.Weight_V = np.random.uniform(low=-scales_variance,high=scales_variance,size=(features,self.units))
            if self.initial_bias == 'zeros' :
                self.bias_q = np.zeros((1,self.units),dtype=np.float32)
                self.bias_k = np.zeros((1,self.units),dtype=np.float32)
                self.bias_v = np.zeros((1,self.units),dtype=np.float32)
            if self.initial_bias == 'random' and self.initial_weight == 'normal' :
                scales_variance = np.sqrt(2/(features + self.units))
                self.bias_q = np.random.normal(loc=0,scale=scales_variance,size=(1,self.units))
                self.bias_k = np.random.normal(loc=0,scale=scales_variance,size=(1,self.units))
                self.bias_v = np.random.normal(loc=0,scale=scales_variance,size=(1,self.units))
            elif self.initial_bias == 'random' and self.initial_weight == 'uniform' :
                scales_variance = np.sqrt(2 / (features + self.units))
                self.bias_q = np.random.uniform(low=-scales_variance,high=scales_variance,size=(1,self.units))
                self.bias_k = np.random.uniform(low=-scales_variance,high=scales_variance,size=(1,self.units))
                self.bias_v = np.random.uniform(low=-scales_variance,high=scales_variance,size=(1,self.units))
            weight_param = (features * self.units) * 3 
            bias_param = self.units * 3 
            self.parameters = weight_param + bias_param
        except Exception as e :
            if self.units is None or self.units <= 0 :
                e.add_note("You have initialization units > 0 => (Attention(units= 1 or more))")
                e.add_note("recomended for 2 exponential values like = > example = 2,4,8,16,32....N")
            if self.initial_weight not in ['normal','uniform'] :
                e.add_note("You must choice initial methode normal or uniform")
            if self.initial_bias not in ['random','zeros'] :
                e.add_note("You must choice initial method random or zeros")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def scaled_dot_product (self,Q,K,V,mask = None) :
        key_dim = K.shape[-1]
        self.__key_dims = key_dim
        scores = np.matmul(Q,K.transpose(0,2,1)) / np.sqrt(key_dim)
        if mask is not None :
            scores += (mask * -1e9)
        scores = self.__softmax(scores)
        self.__scores = scores
        result = np.matmul(scores,V)
        return result,scores
    
    def __call__ (self,Query,Keys,Values) :
        
        try :
            if len(Query.shape) != 3 :
                raise RuntimeError("Query shape must be 3 dims (batch,sequence length,dimention layers)")
            elif len(Keys.shape) != 3 :
                raise RuntimeError("Keys shape must be 3 dims (batch,sequence length,dimention layers)")
            elif len(Values.shape) != 3 :
                raise RuntimeError("Values shape must be 3 dims (batch,sequence length,dimention layers)")
            self.__input = [Query,Keys,Values]
            if self.Weight_Q is None or self.bias_q is None \
                or self.Weight_K is None or self.bias_k is None \
                or self.Weight_V is None or self.bias_v is None :
                    self.__build_weight(Query.shape[-1])
            self.__cache = list()
            Q = np.dot(Query,self.Weight_Q) + self.bias_q
            self.__cache.append(Q)
            K = np.dot(Keys,self.Weight_K) + self.bias_k
            self.__cache.append(K)
            V = np.dot(Values,self.Weight_V) + self.bias_v 
            self.__cache.append(V)
            masked = None 
            if self.Masking :
                masked = self.__create_masking(Q.shape[1])
                masked = np.expand_dims(masked,axis=0)
                self.__mask_vals = masked
            outputs,attention_scores = self.scaled_dot_product(Q,K,V,mask=masked)
            if self.return_attn_state :
                return outputs,attention_scores
            return outputs
        except Exception as e :
            if len(Query.shape) != 3 :
                e.add_note(f"look at : Query shape : {Query.shape} is 3 dims ?" )
            if len(Keys.shape) != 3 :
                e.add_note(f"look at : Keys shape : {Keys.shape} is 3 dims ?")
            if len(Values.shape) !=3 :
                e.add_note(f"look at : Values shape : {Values.shape} is 3 dims ?")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def BackwardPass (self,grad_out) :
        try :
            if self.derivative_mode not in ['jacobian','alternative'] :
                raise RuntimeError("Derivative Mode not availabel")
            d_Values = np.matmul(self.__scores.transpose(0,2,1),grad_out) 
            d_scores = np.matmul(grad_out,self.__cache[-1].transpose(0,2,1))
            if self.derivative_mode == 'jacobian' :
                d_softmax = self.__derivative_softmax(self.__scores)
                d_softmax= np.einsum("bsij,bsj->bsi",d_softmax,d_scores)
            elif self.derivative_mode == 'alternative' :
                d_softmax = self.__alternative_Derivative_softmax(self.__scores,d_scores)
            if self.Masking is True :
                d_softmax *= (1-self.__mask_vals)
            d_log_q = np.matmul(d_softmax,self.__cache[-2]) / self.__key_dims
            d_log_k = np.matmul(d_softmax,self.__cache[-3]) / self.__key_dims
            grad_wq = np.matmul(self.__input[0].transpose(0,2,1),d_log_q)
            grad_wk = np.matmul(self.__input[1].transpose(0,2,1),d_log_k)
            grad_wv = np.matmul(self.__input[2].transpose(0,2,1),d_Values)
            grad_b_q = np.sum(d_log_q,axis=1)
            grad_b_k = np.sum(d_log_k,axis=1)
            grad_b_v = np.sum(d_Values,axis=1)
            grad_b_q = np.mean(grad_b_q,axis=0,keepdims=True,dtype=np.float32)
            grad_b_k = np.mean(grad_b_k,axis=0,keepdims=True,dtype=np.float32)
            grad_b_v = np.mean(grad_b_v,axis=0,keepdims=True,dtype=np.float32)
            grad_out = np.dot(d_log_q,self.Weight_Q.T) + np.dot(d_log_k,self.Weight_K.T) + np.dot(d_Values,self.Weight_V.T)
            return {
                "grad_z" : grad_out,
                "grad_Wq" : grad_wq,
                "grad_wk" : grad_wk,
                "grad_wv" : grad_wv,
                "grad_b_q" : grad_b_q,
                "grad_b_k" : grad_b_k,
                "grad_b_v" : grad_b_v
            }
        except Exception as e :
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

class MultiHeadAttention :
    def __init__ (self,units,num_heads,output_dim = None,use_casual_mask=False,return_attention_state=False,epsilon=1e-9,
                  initial_weight : Literal['normal','uniform'] = 'normal',
                  initial_bias : Literal ['zeros','random'] = 'zeros',
                  derivative_func : Literal['jacobian','alternative']  = 'alternative') :
        self.units = units 
        self.parameters = 0 
        self.num_heads = num_heads
        self.key_dims = units // num_heads
        self.epsilon = epsilon 
        self.weight_q = None 
        self.weight_k = None 
        self.weight_v = None 
        self.weight_o = None 
        self.bias_q = None 
        self.bias_k = None 
        self.bias_v = None 
        self.bias_o = None 
        self.__cache = None 
        self.use_casual_mask = use_casual_mask
        self.return_attention = return_attention_state 
        self.outputdim = output_dim
        self.initial_weight = initial_weight 
        self.initial_bias = initial_bias 
        self.derivative_mode = derivative_func
        self.__input = None 
        self.__scores = None 
        self.__Attention_vals = None 
        self.__shape = None 
        self.__Masking_values = None 
    def __build_weight (self,features) :
        try:
            if self.units <= 0 or self.units is None :
                raise RuntimeError("0 / None  is disagreed ")
            if self.initial_weight not in ['normal','uniform'] :
                raise RuntimeError("initial methode not availabel")
            if self.initial_bias not in ['zeros','random'] :
                raise RuntimeError("initial methode not availabel")
            if self.outputdim < 0  :
                raise RuntimeError("dimention out < 0 is not agreed")
            normal_xavier = np.sqrt(2/(features + self.units))
            uniform_xavier = np.sqrt(6/(features + self.units))
            if self.outputdim is None :
                self.outputdim = 0 
            normal_wo = np.sqrt(2/(self.outputdim + self.units))
            uniform_wo = np.sqrt(6/(self.outputdim + self.units))
            if self.initial_weight == 'normal' :
                self.weight_q = np.random.normal(loc=0,scale=normal_xavier,size=(features,self.units))
                self.weight_k = np.random.normal(loc=0,scale=normal_xavier,size=(features,self.units))
                self.weight_v = np.random.normal(loc=0,scale=normal_xavier,size=(features,self.units))
                if self.outputdim > 0 :
                    self.weight_o = np.random.normal(loc=0,scale=normal_wo,size=(self.units,self.outputdim))
                else : 
                    self.weight_o = np.random.normal(loc=0,scale=normal_wo,size=(self.units,self.units))
            if self.initial_weight == 'uniform' :
                self.weight_q = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(features,self.units))
                self.weight_k = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(features,self.units))
                self.weight_v = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(features,self.units))
                if self.outputdim > 0 :
                    self.weight_o = np.random.uniform(low=-uniform_wo,high=uniform_wo,size=(self.units,self.outputdim))
                else :
                    self.weight_o = np.random.uniform(low=-uniform_wo,high=uniform_wo,size=(self.units,self.units))
            if self.initial_bias == 'zeros' :
                self.bias_q = np.zeros((1,self.units))
                self.bias_k  = np.zeros((1,self.units))
                self.bias_v = np.zeros((1,self.units))
                if self.outputdim == 0 :
                    self.bias_o = np.zeros((1,self.units))
                else :
                    self.bias_o = np.zeros((1,self.outputdim))
            if self.initial_bias == 'random' and self.initial_weight == 'normal' :
                self.bias_q = np.random.normal(loc=0,scale=normal_xavier,size=(1,self.units))
                self.bias_k = np.random.normal(loc=0,scale=normal_xavier,size=(1,self.units))
                self.bias_v = np.random.normal(loc=0,scale=normal_xavier,size=(1,self.units))
                if self.outputdim > 0 :
                    self.bias_o = np.random.normal(loc=0,scale=normal_wo,size=(1,self.outputdim))
                else :
                    self.bias_o = np.random.normal(loc=0,scale=normal_wo,size=(1,self.units))
            if self.initial_bias == 'random' and self.initial_weight == 'uniform' : 
                self.bias_q = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(1,self.units))
                self.bias_k = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(1,self.units))
                self.bias_v = np.random.uniform(low=-uniform_xavier,high=uniform_xavier,size=(1,self.units))
                if self.outputdim > 0 :
                    self.bias_o = np.random.uniform(low=-uniform_wo,high=uniform_wo,size=(1, self.outputdim))
                else :
                    self.bias_o = np.random.uniform(low=-uniform_wo,high=uniform_wo,size=(1,self.units))
            weight_param = (features * self.units) * 3
            bias_param = self.units * 4 
            self.parameters = weight_param + bias_param
        except Exception as e :
            if self.units <=0 or self.units is None :
                e.add_note("You have initialization units > 0 => (MultiheadAttention(units= 1 or more))")
                e.add_note("recomended for 2 exponential values like = > example = 2,4,8,16,32....N")
            elif self.outputdim < 0 :
                e.add_note("for output dimention you can give it to 0 or if you want use it give this parameters. example => 16,32,64...etc")
            if self.initial_weight not in ['normal','uniform'] :
                e.add_note("this layers just support normal or uniform initial weight")
            if self.initial_bias not in ['zeros','random'] :
                e.add_note("this layers just support zeros or random initial bias")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

    def __softmax(self,x) :
        x_max = np.max(x,keepdims=True,axis=-1)
        x_exp = np.exp(x - x_max)
        x_sum = np.sum(x_exp,keepdims=True,axis=-1)
        x_sum[x_sum==0] = self.epsilon
        return x_exp / x_sum
    
    def __derivative_softmax (self,x) :
        batch,head,seq,dim = x.shape
        grad = np.zeros((batch,head,seq,dim,dim))
        for b in range(batch) :
            for h in range(head) :
                for s in range(seq) :
                    y = x[b,h,s].reshape(-1,1)
                    jacob = np.diagflat(y) - np.dot(y,y.T)
                    grad[b,h,s] = jacob 
        return grad 
    
    def __alternative_derivative_softmax (self,softmax_out,gradient) :
        step1 = np.sum(softmax_out * gradient,axis=-1,keepdims=True)
        step2 = softmax_out * (gradient - step1) 
        return step2 
    
    def __create_look_ahead_masking (self,size) :
        mask = 1 - np.tril(np.ones(shape=(size,size),dtype=np.float32))
        return mask[np.newaxis,np.newaxis,:,:]
    
    def __splitheads (self,x) :
        batch_size = x.shape[0]
        sequence_lenght = x.shape[1]
        x = np.reshape(x,newshape=(batch_size,sequence_lenght,self.num_heads,(self.units // self.num_heads)))
        x = np.transpose(x,axes=[0,2,1,3])
        return x 
    
    def scaled_dot_product (self,Q,K,V,mask=None) :
        scores = np.matmul(Q,K.transpose(0,1,3,2)) / np.sqrt(self.key_dims)
        if mask is not None :
            scores += (mask * -1e9)
        scores = self.__softmax(scores)
        self.__scores = scores
        attention = np.matmul(scores,V)
        return attention,scores
    
    def __call__ (self,Query,Keys,Values) :
        try :
            self.__input = [Query,Keys,Values]
            if len(Query.shape) != 3 :
                raise RuntimeError(f"Error shape Query : {len(Query.shape)}")
            elif len(Keys.shape) != 3 :
                raise RuntimeError(f"Error shape Keys : {len(Keys.shape)}")
            elif len(Values.shape) != 3 :
                raise RuntimeError(f"Error shape Values : {len(Values.shape)}")
            
            if self.weight_q is None or self.bias_q is None\
            or self.weight_k is None or self.bias_k is None\
            or self.weight_v is None or self.bias_q is None\
            or self.weight_o is None or self.bias_o is None :
                self.__build_weight(Query.shape[-1])

            self.__cache = list()
            Q = np.dot(Query,self.weight_q) + self.bias_q
            K = np.dot(Keys,self.weight_k) + self.bias_k
            V = np.dot(Values,self.weight_v ) + self.bias_v
            Q = self.__splitheads(Q)
            K = self.__splitheads(K)
            V = self.__splitheads(V)
            self.__shape = Q.shape
            self.__cache.append(Q)
            self.__cache.append(K)
            self.__cache.append(V)
            masked = None 
            if self.use_casual_mask :
                masked = self.__create_look_ahead_masking(Query.shape[-2])
                masked = np.expand_dims(masked,axis=0)
                self.__Masking_values = masked
            
            attention,scores = self.scaled_dot_product(Q,K,V,mask=masked)
            attention = np.transpose(attention,axes=[0,2,1,3])
            attention = np.reshape(attention,newshape=(Query.shape[0],Query.shape[1],self.units))
            result = np.dot(attention,self.weight_o) + self.bias_o
            self.__Attention_vals = attention
            self.__cache.append(result)
            if self.return_attention :
                return result,scores
            return result
        except Exception as e :
            e.add_note("Input must be 3 dims ")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def BackwardPass (self,gradient) :
        try:
            if self.derivative_mode not in ['jacobian','alternative'] :
                raise RuntimeError("Derivative mode not vailabel")
            batch,seq,dim = self.__shape
            d_Wo = np.matmul(self.__Attention_vals.transpose(0,2,1),gradient)
            d_attention_f_W_o = np.dot(gradient,self.weight_o.T)
            d_wv = np.matmul(self.__scores.transpose(0,1,3,2),d_attention_f_W_o)
            d_scores = np.matmul(self.__cache[-1].transpose(0,1,3,2),d_attention_f_W_o)
            if self.derivative_mode == 'jacobian' :
                grad_softmax = self.__derivative_softmax(self.__scores)
                d_softmax = np.einsum("bhsdj,bhsj->bhsd",grad_softmax,d_scores)
            elif self.derivative_mode == 'alternative' :
                d_softmax = self.__alternative_derivative_softmax(self.__scores,d_scores)
            if self.use_casual_mask is True :
                d_softmax *= (1 - self.__Masking_values)
            d_q = np.matmul(d_softmax,self.__cache[-2]) / np.sqrt(self.key_dims)
            d_k = np.matmul(d_softmax,self.__cache[-3]) / np.sqrt(self.key_dims)
            d_q = np.transpose(d_q,(0,2,1,3))
            d_k = np.transpose(d_k,(0,2,1,3))
            d_v = np.transpose(d_wv,(0,2,1,3))
            grad_q = np.matmul(self.__input[0].transpose(0,2,1),d_q.reshape(batch,seq,dim))
            grad_k = np.matmul(self.__input[1].transpose(0,2,1),d_k.reshape(batch,seq,dim))
            grad_v = np.matmul(self.__input[2].transpose(0,2,1),d_v.reshape(batch,seq,dim))
            grad_b_q = np.sum(d_q,axis=1)
            grad_b_k = np.sum(d_k,axis=1)
            grad_b_v = np.sum(d_v,axis=1)
            grad_b_o = np.sum(d_Wo,axis=1)
            grad_b_q = np.mean(grad_b_q,axis=0,keepdims=True,dtype=np.float32)
            grad_b_k = np.mean(grad_b_k,axis=0,keepdims=True,dtype=np.float32)
            grad_b_v = np.mean(grad_b_v,axis=0,keepdims=True,dtype=np.float32)
            grad_b_o = np.mean(grad_b_o,axis=0,keepdims=True,dtype=np.float32)
            grad_nq = np.dot(grad_q,self.weight_q.T)
            grad_nk = np.dot(grad_k,self.weight_k.T)
            grad_nv = np.dot(grad_v,self.weight_v.T)
            grad_next = grad_nq + grad_nk + grad_nv 
            return {
                "grad_z" :grad_next,
                "grad_wq" : grad_q,
                "grad_wk" : grad_k,
                "grad_wv" : grad_v,
                "grad_wo" :d_Wo,
                "grad_b_q" : grad_b_q,
                "grad_b_k" : grad_b_k,
                "grad_b_v" : grad_b_v,
                "grad_b_o" : grad_b_v
            }
        except Exception as e :
            e.add_note("choice jacobian or alternative")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 


class Embedding :
    def __init__ (self,input_dim,output_dim) :
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = None 
        self.parameters = 0
    
    def __build_weight(self) :
        try :
            if self.input_dim == 0 or self.output_dim is None\
            or self.input_dim is None or self.output_dim is None :
                raise ValueError("0 units is disagreed for layers") 
            self.weight = np.random.rand(self.input_dim,self.output_dim) * 0.1
            self.parameters = self.input_dim * self.output_dim
        except Exception as e :
            e.add_note("Units must initilization => Embedding (units)")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def __call__ (self,x) :
        if self.weight is None :
            self.__build_weight()
        return self.weight[x]
    
class SimpleRNN :
    def __init__ (self,units,return_hidden = False) :
        self.units = units 
        self.return_hidden = return_hidden
        self.weight_sequence = None 
        self.weight_hidden = None 
        self.bias = None 
        self.h = None 
    
    def __build_weight (self,features) :
        try : 
            if self.units == 0 or self.units is None :
                raise ValueError("0 is disagreed for layers")
            scales_Variance = np.sqrt(2 / features)
            self.weight_sequence = np.random.normal(loc=0,scale=scales_Variance,size=(features,self.units))
            self.weight_hidden = np.random.normal(loc=0,scale=scales_Variance,size=(self.units,self.units))
            self.bias = np.zeros((1,self.units))
            self.h = np.zeros((1,self.units))
        except Exception as e :
            e.add_note("You must initialization units for this layers => SimpleRNN(units)")
            e.add_note("Recomended Units Values is 2 exponential => 2,4,6,16,32,...N")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

    def __step_excetion (self,x) :

        if self.units == 0 or self.units is None :
            raise ValueError("0 is disagreed for layers")
        sequence_logits = np.dot(x,self.weight_sequence) 
        hidden_logits = np.dot(self.h,self.weight_hidden)
        logits = (sequence_logits + hidden_logits) + self.bias 
        self.h = np.tanh(logits)
        return self.h 
    
    def __call__ (self,x) :
        if self.weight_sequence is None or self.weight_hidden is None :
            self.__build_weight(x.shape[-1])
        seq_out = list()
        for iter in range(x.shape[0]) :
            x_iter = x[iter:iter+1]
            self.h = self.__step_excetion(x_iter)
            seq_out.append(self.h)
        seq_out = np.vstack(seq_out,dtype=np.float32)
        if self.return_hidden :
            return seq_out,self.h 
        return seq_out

class LSTM :
    def __init__ (self,units,return_state=False,return_sequence=False,weight_initial : Literal['normal','uniform'] = 'normal',
                  bias_initial : Literal ['zeros','random'] = 'zeros') :
        self.units = units 
        self.return_sequence = return_sequence
        self.return_state = return_state
        self.weight_forgot_gate = None 
        self.weight_new_information_gate = None 
        self.weight_output_gate = None 
        self.weight_cell_state = None 
        self.bias_fg = None 
        self.bias_new_in = None 
        self.bias_output = None 
        self.bias_cell = None 
        self.cell_state = None 
        self.hidden_state = None 
        self.parameters = None 
        self.weight_initial = weight_initial
        self.bias_initial = bias_initial
    
    def __sigmoid (self,x) :
        return 1 / (1 + np.exp(-x))
    
    def __build_weight (self,features) :
        try :
            features += self.units
            if self.units is None or self.units == 0:
                raise ValueError ("0 / None units is disagreed for layers")
            try :
                if self.weight_initial not in ['normal','uniform'] :
                    raise ValueError("The initial methods not available at layers")
                if self.weight_initial == 'normal' :
                    normal_variance = np.sqrt(2/(features + self.units))
                    self.weight_forgot_gate = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
                    self.weight_new_information_gate = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
                    self.weight_output_gate = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
                    self.weight_cell_state = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
                elif self.weight_initial == 'uniform' :
                    uniform_variance = np.sqrt(6 / (features + self.units))
                    self.weight_forgot_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(features,self.units))
                    self.weight_new_information_gate = np.random.uniform(low=-uniform_variance,high = uniform_variance,size=(features,self.units))
                    self.weight_output_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(features,self.units))
                    self.weight_cell_state = np.random.uniform (low=-uniform_variance,high=uniform_variance,size=(features,self.units))
                    try :
                        if self.bias_initial not in ['zeros','random'] :
                            raise ValueError("bias initial not available at layers")
                        if self.bias_initial == 'zeros' :
                            self.bias_fg = np.zeros((1,self.units))
                            self.bias_new_in = np.zeros(1,self.units)
                            self.bias_cell = np.zeros(1,self.units)
                            self.bias_output = np.zeros(1,self.units)
                        if self.bias_initial == 'random' and self.weight_initial == 'normal' :
                            self.bias_fg = np.random.normal(loc=0,scale=normal_variance,size=(1,self.units))
                            self.bias_new_in = np.random.normal(loc=0,scale=normal_variance,size=(1,self.units))
                            self.bias_output = np.random.normal(loc=0,scale=uniform_variance,size=(1,self.units))
                            self.bias_cell = np.random.normal(loc=0,scale=uniform_variance,size=(1,self.units))
                        elif self.bias_initial == 'random' and self.weight_initial == 'uniform' :
                            self.bias_fg = np.random.uniform(los=-uniform_variance,high=uniform_variance,size=(1,self.units))
                            self.bias_new_in = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
                            self.bias_output = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
                            self.bias_cell = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
                    except Exception as e :
                        e.add_note("Choice the initial (zeros) ot (random)")
                        traceback.print_exception(type(e),e,e.__traceback__)
                        raise 
            except  Exception as e :
                e.add_note("Choice the initial (normal) or (uniform)")
                traceback.print_exception(type(e),e,e.__traceback__)
                raise 
        except Exception as e :
            e.add_note("You must initialization units for this layers => LSTM(units)")
            e.add_note("Recomended Units Values is 2 exponential => 2,4,6,16,32,...N")  
            traceback.print_exception(type(e),e,e.__traceback__)
            raise    

    def __step_excution (self,x_time_step,hidden_time_step,cell_time_step) :
        combined_input_hidden = np.concatenate([x_time_step,hidden_time_step],axis=-1,dtype=np.float32)
        f_g_logits = np.dot(combined_input_hidden,self.weight_forgot_gate) + self.bias_fg
        info_g_logits = np.dot(combined_input_hidden,self.weight_new_information_gate) + self.bias_new_in 
        out_g_logits = np.dot(combined_input_hidden,self.weight_output_gate) + self.bias_output
        cell_logits = np.dot(combined_input_hidden,self.weight_cell_state) + self.bias_cell 
        
        forgot_gate = self.__sigmoid(f_g_logits)
        input_info = self.__sigmoid(info_g_logits)
        output_gate = self.__sigmoid(out_g_logits)
        cell_info_candidate = np.tanh(cell_logits,dtype=np.float32)

        cell_state = forgot_gate * cell_time_step + input_info * cell_info_candidate 
        hidden_state = output_gate * np.tanh(cell_state)
        return hidden_state,cell_state
    
    def __call__ (self,x,initial_state=None) : 
        batch_size,sequence,d_model = x.shape
        if self.weight_forgot_gate is None or self.weight_new_information_gate is None\
        or self.weight_output_gate is None or self.weight_cell_state is None :
            self.__build_weight(d_model)
            if initial_state is not None :
                hidden_state,cell_state = initial_state
                try :
                    if hidden_state.shape[-1] != self.weight_forgot_gate.shape[-1] or cell_state.shape[-1] != self.weight_cell_state.shape[-1] :
                        raise ValueError(f"Mist Shape at {hidden_state.shape[-1]} to {self.weight_cell_state.shape[-1]}")
                    self.hidden_state = hidden_state
                    self.cell_state = cell_state
                except Exception as e :
                    e.add_note("hidden/cell state dimention must have same dimention ==> h/c d (64) = layers unit (64)")
                    traceback.print_exception(type(e),e,e.__traceback__)
                    raise 
            self.hidden_state = np.zeros((batch_size,self.units))
            self.cell_state = np.zeros((batch_size,self.units))
        outputs = list()
        for t in range(sequence) :
            x_t = x[:,t,:]
            self.hidden_state,self.cell_state = self.__step_excution(x_time_step=x_t,hidden_time_step=self.hidden_state,
                                                                     cell_time_step=self.cell_state)
            outputs.append(self.hidden_state)
        
        logits_out = np.stack(outputs,axis=1)
        if self.return_sequence :
            return logits_out
        if self.return_state :
            return logits_out[:,-1,:],self.hidden_state,self.cell_state
        if self.return_sequence and self.return_state :
            return logits_out,self.hidden_state,self.cell_state
        
        return logits_out[:,-1,:]

class GRU :
    def __init__(self,units,return_sequence = False,return_hidden_state = False,
                 initial_weight : Literal['normal','uniform'] = 'normal',
                 initial_bias : Literal['zeros','random'] = 'zeros',
                 use_gradient_reflector : Literal[True,False] = True) :
        self.units = units 
        self.initial_weight = initial_weight
        self.initial_bias = initial_bias
        self.return_sequence = return_sequence
        self.return_hidden_state = return_hidden_state
        self.use_engine_grad = use_gradient_reflector
        self.weight_up_gate = None 
        self.weight_re_gate = None 
        self.weight_candidate_gate = None 
        self.bias_up_gate = None 
        self.bias_re_gate = None 
        self.bias_candidate_gate = None  
        self.parameter = None 
        self.hidden_state = None
    
    def __sigmoid (self,x) :
        return 1 / (1 + np.exp(-1))
    
    def __derivative_sigmoid (self,x) :
        return x * (1 - x)
    
    def __build_weight (self,features) : 
        try:
            if self.initial_weight not in ['normal','uniform'] :
                raise RuntimeError("initial methode just availabel for normal and uniform")
            if self.initial_bias not in ['random','zeros'] :
                raise RuntimeError("initial bias methode just availabel for zeros and random")
            if self.units <= 0 or self.units is None  :
                raise RuntimeError("The units can't if is 0 or None")
            features += self.units
            normal_variance = np.sqrt(2 / (features + self.units))
            uniform_variance = np.sqrt(6 / (features + self.units))
            if self.initial_weight == 'normal':
                self.weight_up_gate = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
                self.weight_re_gate = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
                self.weight_candidate_gate = np.random.normal(loc=0,scale=normal_variance,size=(features,self.units))
            elif self.initial_weight == 'uniform' :
                self.weight_up_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(features,self.units))
                self.weight_re_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(features,self.units))
                self.weight_candidate_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(features,self.units))
            if self.initial_bias == 'random' and self.initial_weight == 'normal' :
                self.bias_up_gate = np.random.normal(loc=0,scale=normal_variance,size=(1,self.units))
                self.bias_re_gate = np.random.normal(loc=0,scale=normal_variance,size=(1,self.units))
                self.bias_candidate_gate = np.random.normal(loc=0,scale=normal_variance,size=(1,self.units))
            elif self.initial_bias == 'random' and self.initial_weight == 'uniform' :
                self.bias_up_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
                self.bias_re_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
                self.bias_candidate_gate = np.random.uniform(low=-uniform_variance,high=uniform_variance,size=(1,self.units))
            if self.initial_bias == 'zeros' :
                self.bias_re_gate = np.zeros((1,self.units))
                self.bias_up_gate = np.zeros((1,self.units))
                self.bias_candidate_gate = np.zeros((1,self.units))
            self.hidden_state = np.zeros((1,self.units))
            param_weight = (features * self.units) * 3 
            param_bias = self.units * 3 
            self.parameter = param_weight + param_bias
        except Exception as e :
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    def __step_execution (self,x_t,h_t) :
        if self.use_engine_grad:
            combine_input = np.concatenate([x_t,h_t],axis=-1)
            if not isinstance(self.weight_up_gate,ll.GradientReflector) :
                self.weight_up_gate = ll.GradientReflector(self.weight_up_gate,_op='update_gate')
            if not isinstance(self.weight_re_gate,ll.GradientReflector) :
                self.weight_re_gate = ll.GradientReflector(self.weight_re_gate,_op='reset_gate')
            if not isinstance(self.weight_candidate_gate,ll.GradientReflector) :
                self.weight_candidate_gate = ll.GradientReflector(self.weight_candidate_gate,_op='candidate_gate')
            if not isinstance(self.bias_up_gate,ll.GradientReflector) :
                self.bias_up_gate = ll.GradientReflector(self.bias_up_gate,_op='bias_update')
            if not isinstance(self.bias_re_gate,ll.GradientReflector) :
                self.bias_re_gate = ll.GradientReflector(self.bias_re_gate,_op='bias_reset')
            if not isinstance(self.bias_candidate_gate,ll.GradientReflector) :
                self.bias_candidate_gate = ll.GradientReflector(self.bias_candidate_gate,_op='bias_candidate')
            logits_up = ll.dot(combine_input,self.weight_up_gate) + self.bias_up_gate
            logits_re = ll.dot(combine_input,self.weight_re_gate) + self.bias_re_gate
            update_gate = ll.activations.sigmoid(logits_up)
            reset_date = ll.activations.sigmoid(logits_re)
