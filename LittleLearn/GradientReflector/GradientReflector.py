import numpy as np 
import traceback          
import matplotlib.pyplot as plt 
import networkx as nx   

class GradientReflector :
    def __init__ (self,tensor,_children=(),_op='') :
        self.tensor = np.array(tensor)
        self.gradient =  np.zeros_like(self.tensor)
        self._backwardpass = lambda : None 
        self._Node = _children
        self._op = _op 
    
    def __repr__(self):
        return (f"(Tensor with shape : ({self.tensor.shape}) : \n  {self.tensor})")
    
    def get_gradient (self) :
        return self.gradient
    
    def __adjust_gradient(self, out_grad, input_grad_shape):
        while out_grad.ndim > len(input_grad_shape) :
            out_grad = np.sum(out_grad,axis=0)
        while out_grad.ndim < len(input_grad_shape) :
            out_grad = np.expand_dims(out_grad,axis=0)
        for i,(out_shape,input_shape) in enumerate(zip(out_grad.shape,input_grad_shape)) :
            if out_shape != input_shape:
                out_grad = np.sum(out_grad,axis=i,keepdims=True)
        return out_grad

    def __add__(self,other) :
        other = other if isinstance(other,GradientReflector) else GradientReflector(other)
        out = GradientReflector((self.tensor + other.tensor),(self,other),'+')
        def _backward () :
            grad = self.__adjust_gradient(out.gradient,self.tensor.shape)
            grad_other = self.__adjust_gradient(out.gradient,other.tensor.shape)
            self.gradient += grad 
            other.gradient += grad_other
        out._backwardpass = _backward
        return out 
    
    def __sub__ (self,other) :
        other = other if isinstance(other,GradientReflector) else GradientReflector(other)
        out = GradientReflector((self.tensor - other.tensor),(self,other),'-')
        def _backward () :
            grad = self.__adjust_gradient(out.gradient,self.gradient.shape)
            self.gradient += np.ones_like(self.tensor) * grad
            other.gradient += (-1 * np.ones_like(self.tensor)) * grad
        out._backwardpass = _backward
        return out 
    
    def pow (self,power_values) :
        out = GradientReflector(np.power(self.tensor,power_values),(self,),'pow')
        def _backward() :
            grad = power_values * (np.power(self.tensor,power_values-1))
            self.gradient += grad * out.gradient
        out._backwardpass = _backward
        return out  
    
    def __mul__ (self,other) :
        other = other if isinstance(other,GradientReflector) else GradientReflector(other)
        out = GradientReflector((self.tensor * other.tensor),(self,other),'*')
        def _backward () :
            grad = self.__adjust_gradient(out.gradient,self.gradient.shape)
            self.gradient += other.tensor * grad
            other.gradient += self.tensor * grad
        out._backwardpass = _backward
        return out 
    
    def __truediv__(self,other) :
        other = other if isinstance(other,GradientReflector) else GradientReflector(other) 
        out = GradientReflector((self.tensor / other.tensor),(self,other),"/")
        def _backward () :
            grad = self.__adjust_gradient(out.gradient,self.gradient.shape)
            self.gradient += (np.ones_like(self.tensor) / other.tensor) * out.gradient
            other.gradient += (-self.tensor / np.power(other.tensor,2)) * out.gradient
        out._backwardpass = _backward
        return out 
    
    def __neg__ (self) :
        return self *-1
    
    def __pow__ (self,other) :
        assert isinstance(other,(int,float))
        out = GradientReflector(np.power(self.tensor,other),(self,),f'**{other}') 
        def _backward() :
            self.gradient += (other * np.power(self.tensor,other -1)) * out.gradient
        out._backwardpass = _backward
        return out 
    
    def __radd__ (self,other) : 
        return self + other 
    
    def __rsub__ (self,other) :
        return self - other 
    
    def __rtruediv__ (self,other) :
        return self * (other**-1)

    def __rmul__ (self,other) :
        return self * other 
    
    def relu (self) :
        out = GradientReflector(np.maximum(0,self.tensor),(self,),'relu')
        def _backward() :
            self.gradient += np.where(self.tensor >0 , 1 ,0) * out.gradient
        out._backwardpass = _backward
        return out 
    
    def leaky_relu(self,alpha) : 
        forward = np.where(self.tensor > 0 , self.tensor,self.tensor * alpha)
        outputs = GradientReflector(forward,(self,),'leaky_relu')
        def _backward () :
            grad = np.where(self.tensor > 0,1,alpha)
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def tanh (self) :
        outputs = GradientReflector((np.tanh(self.tensor)),(self,),'tanh')
        def _backward() :
            self.gradient = (1 - np.power(np.tanh(self.tensor),2)) * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def swish (self,Beta=1) :
        shifted_1 = 1 / (1 + np.exp(-(Beta * (self.tensor))))
        s_result = self.tensor * shifted_1
        outputs = GradientReflector(s_result,(self,),'swish')
        def _backward() :
            shift1 = s_result
            shift2 = 1 / (1 + np.exp(-self.tensor))
            shift3 = 1 - s_result
            grad = shift1 * shift2 & shift3
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def gelu(self) :
        coef = 1.702
        def sigmoid (x) :
            return 1 / (1 + np.exp(-x))
        sig_values = sigmoid(coef * self.tensor)
        result = self.tensor * sig_values
        outputs = GradientReflector(result,(self,),'gelu') 
        def _backward () :
            der_sig = sig_values * (1 - sig_values)
            grad  = sig_values + self.tensor * der_sig * coef
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def linear (self) :
        outputs = GradientReflector(self.tensor,(self,),'linear')
        def _backward () :
            self.gradient += self.tensor * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def softmax (self,y_label=None,axis=None,keepdims=False,epsilon=1e-6) :
        try :
            if axis == -1 and keepdims is False :
                raise RuntimeError("if use axis -1 must keepdims = True")
            x_max = np.max(self.tensor,axis=axis,keepdims=keepdims)
            x_exp = np.exp(self.tensor - x_max)
            x_sum = np.sum(x_exp,axis=axis,keepdims=keepdims)
            if len(x_sum.shape) == 0 :
                x_sum = x_sum
            else :
                x_sum[x_sum==0] += epsilon
            prob = x_exp / x_sum
            outputs = GradientReflector(prob,(self,),'softmax')
            def _backward () :
                if y_label is None :
                    step1 = np.sum(outputs.gradient - prob,axis=axis,keepdims=keepdims)
                    step2 = prob - (outputs.gradient - step1)
                    self.gradient += step2 
                else :
                    check = list()
                    for i in range(len(y_label)) :
                        check.append(prob[i] - y_label[i])
                    grad = np.vstack(check,dtype=np.float64)
                    self.gradient += grad * outputs.gradient
            outputs._backwardpass = _backward
            return outputs 
        except Exception as e :
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
        
    def exp(self) :
        outputs = GradientReflector((np.exp(self.tensor)),(self,),'exp')
        def _backward() :
            self.gradient = np.exp(self.tensor) * outputs.gradient 
        outputs._backwardpass = _backward
        return outputs
    
    def log(self) :
        outputs = GradientReflector((np.log(self.tensor)),(self,),'log')
        def _backward() :
            self.gradient += (1 / self.tensor) * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def matmul (self,other,transpose_a=False,transpose_b = False) :
        other = other if isinstance(other,GradientReflector) else GradientReflector(other)
        a_values = self.tensor.swapaxes(-1,-2) if transpose_a else self.tensor
        b_values = other.tensor.swapaxes(-1,-2) if transpose_b else other.tensor
        result = np.matmul(a_values,b_values)
        outputs = GradientReflector(result,(self,other),'matmul')
        def _backward () :
            if transpose_a :
                grad_a = np.matmul(outputs.gradient,b_values.swapaxes(-1,-2))
                grad_b = np.matmul(a_values.swapaxes(-1,-2),outputs.gradient)
                grad_a = np.swapaxes(grad_a,-1,-2)
                self.gradient = grad_a
                other.gradient = grad_b 
            elif transpose_b :
                grad_a = np.matmul(outputs.gradient,b_values.swapaxes(-2,-1))
                grad_b = np.matmul(a_values.swapaxes(-1,-2),outputs.gradient)
                self.gradient = grad_a
                other.gradient = np.swapaxes(grad_b,-1,-2)
            else :
                self.gradient = np.matmul(outputs.gradient,other.tensor.swapaxes(-1,-2))
                other.gradient = np.matmul(self.tensor.swapaxes(-1,-2),outputs.gradient)
        outputs._backwardpass = _backward
        return outputs 
    
    def dot (self,other) :
        other = other if isinstance(other,GradientReflector) else GradientReflector(other)
        outputs = GradientReflector(np.dot(self.tensor,other.tensor),(self,other),'dot')
        
        def _backward() :
            self.gradient = np.dot(outputs.gradient,other.tensor.T)
            other.gradient = np.dot(self.tensor.T,outputs.gradient)
        outputs._backwardpass = _backward
        return outputs
    
    def sin(self) :
        outputs = GradientReflector(np.sin(self.tensor),(self,),'sin')
        def _backward() :
            self.gradient += np.cos(self.tensor) * outputs.gradient
        
        outputs._backwardpass = _backward
        return outputs 
    
    def cos (self) :
        outputs = GradientReflector(np.cos(self.tensor),(self,),'cos')
        def _backward() :
            self.gradient += np.sin(self.tensor) * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def tan(self) :
        outputs = GradientReflector(np.tan(self.tensor),(self,),'tan')
        def _backward() :
            self.gradient += (1 / np.power(np.cos(self.tensor),2)) * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def clip (self,min_vals,max_vals) :
        outputs = GradientReflector(np.clip(self.tensor,min_vals,max_vals),(self,),'clip')
        def _backward() :
            grad = (self.tensor >= min_vals) & (self.tensor <= max_vals)
            self.gradient += grad.astype(float) * outputs.gradient
        outputs._backwardpass = _backward
        return outputs 
    
    def __getitem__(self,idx) :
        outputs = GradientReflector(self.tensor[idx],(self,),f'getitem{idx}')
        def _backward () :
            grad = np.zeros_like(self.tensor)
            grad[idx] = outputs.gradient 
            self.gradient += grad
        outputs._backwardpass = _backward
        return outputs 
    
    def reshape(self,shape=()) :
        outputs = GradientReflector(np.reshape(self.tensor,shape),(self,),'reshape')
        def _backward () :
            self.gradient += np.reshape(outputs.tensor,(self.tensor.shape))
        outputs._backwardpass = _backward
        return outputs 
    
    def sigmoid(self) :
        result = 1 / (1 + np.exp(-self.tensor))
        outputs = GradientReflector(result,(self,),'sigmoid')
        def _backward () :
            grad = result * (1 - result)
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs 
    

    def binarycrossetnropy (self,y_true,epsilon=1e-6) :
        n = len(y_true) 
        y_pred = self.tensor
        y_pred = np.clip(y_pred,epsilon,1-epsilon)
        loss = (-1/n) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        outputs = GradientReflector(loss,(self,),'binary_crossentropy')
        def _backward () :
            grad = (1 /n ) * ((y_pred - y_true)/(y_pred * (1 - y_pred)))
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs 
    
    def categoricallcrossentropy (self,y_true,epsilon=1e-6) :
        y_pred = np.clip(self.tensor,epsilon,1-epsilon)
        loss = -np.sum((y_true) * np.log(y_pred))
        outputs = GradientReflector(loss,(self,),'categoricall_crossentropy')
        def _backward () :
            grad = (y_pred - y_true) / len(y_true) 
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs 
    
    def sparsecategoricallcrossentropy (self,y_true,epsilon=1e-6) :
        y_pred = self.tensor
        y_pred = np.clip(y_pred,epsilon,1 - epsilon)
        labels_indeks_True = np.arange(len(y_true))
        Loss = -np.log(y_pred[labels_indeks_True,y_true])
        out = np.mean(Loss)
        outputs = GradientReflector(out,(self,),'sparse_categoricallcrossentropy')
        def _backward() :
            grad = y_pred.copy()
            grad[labels_indeks_True,y_true] -= 1
            grad_ = grad / len(y_true)
            self.gradient = grad_ * outputs.gradient
        outputs._backwardpass = _backward
        return outputs 
    
    def meansquareerror (self,y_true) :
        y_pred = self.tensor
        loss = np.mean(np.power((y_pred - y_true),2))
        outputs = GradientReflector(loss,(self,),'mse')
        def _backward () :
            grad = (2/len(y_true)) * (y_pred - y_true)
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs

    def meanabsoluteerror (self,y_true) :
        y_pred = self.tensor
        loss = np.mean(np.abs(y_pred - y_true))
        outputs = GradientReflector(loss,(self,),'mae')
        def _backward() :
            grad =  (1/(len(y_true))) * np.sign(y_pred - y_true)
            self.gradient += grad * outputs.gradient
        outputs.backwardpass = _backward
        return outputs
    
    def hubber_loss (self,y_true,delta=1.0) : 
        y_pred = self.tensor
        loss = y_pred - y_true
        hub_loss = np.where(np.abs(loss) <= delta,0.5 * np.mean(np.power(loss,2)),
                            delta * (np.abs(loss) - (0.5 * delta)))
        hub_loss = np.mean(hub_loss)
        outputs = GradientReflector(hub_loss,(self,),'huber_lost')
        def _backward() :
            grad = np.where(np.abs(loss) <= delta,loss,delta * np.sign(loss))
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs 

    def sum(self,axis=None,keepdims=False) :
        values = np.sum(self.tensor,axis=axis,keepdims=keepdims)
        outputs = GradientReflector(values,(self,),'sum')
        def _backward () :
            self.gradient += np.ones_like(self.tensor) * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def transpose(self,shape=()) :
        shape = shape 
        tensor = self.tensor
        outputs = GradientReflector(np.transpose(tensor,axes=shape),(self,),'transpose')
        def _backward () :
            self.gradient = np.transpose(outputs.gradient,shape)
        outputs._backwardpass = _backward
        return outputs
    
    def max(self,axis=None,keepdims=False) :
        out = np.max(self.tensor,axis=axis,keepdims=keepdims)
        outputs = GradientReflector(out,(self,),'max')
        def _backward() :
            masked = (out == self.tensor)
            grad = outputs.gradient 
            if axis is not None :
                grad = np.expand_dims(grad,axis=axis)
            maksed_grad = masked * grad
            self.gradient = maksed_grad
        outputs._backwardpass = _backward
        return outputs 
    
    def variace(self,axis=None,keepdims=False) :
        x = self.tensor 
        means = np.mean(x,axis=axis,keepdims=keepdims) 
        var = np.mean(np.power((x - means),2),axis=axis,keepdims=keepdims)
        outputs = GradientReflector(var,(self,),'variance')
        def _backward() :
            grad =  (2/len(x[axis])) * (x - means)
            self.gradient += grad * outputs.gradient
        outputs._backwardpass = _backward
        return outputs
    
    def std(self,axis=None,keepdims=False) :
        x = self.tensor
        mean = np.sum(x,axis=axis,keepdims=keepdims) / len(x[axis])
        var = np.mean(np.power((x - mean),2),axis=axis,keepdims=keepdims)
        std_vals = np.sqrt(var)
        outputs = GradientReflector(std_vals,(self,),'std')
        def _backward() :
            grad = (x - mean) / (len(x) * std_vals)
            self.gradient += grad 
        outputs._backwardpass = _backward
        return outputs

    def layernormalization_backend(self,gamma,beta,epsilon=1e-6) :
        gamma = gamma if isinstance(gamma,GradientReflector) else GradientReflector(gamma)
        beta = beta if isinstance(beta,GradientReflector) else GradientReflector(beta)
        mean = np.sum(self.tensor,axis=-1,keepdims=True) / len(self.tensor[-1])
        variance = np.mean(((self.tensor - mean)**2),axis=-1,keepdims=True)
        part_1 = self.tensor - mean 
        part_2 = variance + epsilon 
        normal_r = part_1 / np.sqrt(part_2)
        out = gamma.tensor * normal_r + beta.tensor
        output = GradientReflector(out,(self,gamma,beta),'layer_normal_backend')
        def _backward () :
            beta.gradient += np.sum(output.gradient,axis=0)
            gamma.gradient += np.sum((output.gradient * normal_r),axis=0)
            d_normal = output.gradient * gamma.tensor 
            d_var = np.sum(d_normal * (self.tensor - mean) * -0.5 * np.power(variance + epsilon,1.5) 
                           ,axis=-1,keepdims=True)
            d_mean = np.sum((d_normal * -1) / np.sqrt(variance + epsilon),axis=-1,keepdims=True) \
            + d_var * np.mean(-2 * (self.tensor - mean),axis=-1,keepdims=True)
            grad = d_normal  / np.sqrt(variance + epsilon) + d_var * 2 * (self.tensor - mean)\
            / self.tensor.shape[-1] + (d_mean / self.tensor.shape[-1])
            self.gradient += grad 
        output._backwardpass = _backward
        return output 
    
    def bacthnormalization_backend(self,gamma,beta,epsilon=1e-6) :
        gamma = gamma if isinstance(gamma,GradientReflector) else GradientReflector(gamma)
        beta = beta if isinstance(beta,GradientReflector) else GradientReflector(beta)
        mean = np.mean(self.tensor,axis=0,keepdims=True)
        var = np.mean(np.power((self.tensor - mean),2),axis=0,keepdims=True)
        step_1 = self.tensor - mean 
        step_2 = var + epsilon
        normal = step_1 / np.sqrt(step_2)
        out = gamma * normal + beta
        outputs = GradientReflector(out,(self,gamma,beta),'batch_normal_backend')
        def _backward() :
            beta.gradient += np.sum(outputs.gradient,axis=0)
            gamma.gradient += np.sum((outputs.gradient * normal),axis=0)
            d_normal = outputs.gradient * gamma.tensor
            d_var = np.sum(d_normal * (self.tensor - mean) * 0.5 * np.power((var + epsilon),1.5),axis=0,keepdims=True)
            d_means = np.sum((d_normal * -1 )/ np.sqrt(var + epsilon),axis=0,keepdims=True)\
            + d_var * np.mean(-2 * (self.tensor - mean),axis=0,keepdims=True)
            grad = d_normal / np.sqrt(var + epsilon)\
            + d_var * 2 * (self.tensor - mean) / self.tensor + (d_means / self.tensor.shape[0])
            self.gradient += grad 
        outputs._backwardpass = _backward
        return outputs

    def get_tensor(self) :
        return self.tensor
    
    @property 
    def shape (self) :
        return self.tensor.shape 

    def backwardpass (self) :
        topo = list()
        visited = set()
        def build_topo (v):        
            if v not in visited :
                visited.add(v)
                for child in v._Node :
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.gradient = np.ones_like(self.tensor)
        for node in reversed(topo) :
            node._backwardpass()
    
    def plot_trace_operation (self) :
        visited = set()
        G = nx.DiGraph()

        def build(Node) :
            if Node not in visited :
                visited.add(Node)
                G.add_node(id(Node),label=Node._op)
                for parent in Node._Node :
                    G.add_edge(id(parent),id(Node))
                    build(parent)

        build(self)
        labels = nx.get_node_attributes(G,'label')
        pos = nx.spring_layout(G)
        nx.draw(G,pos,with_labels=True,labels=labels,node_color = 'lightblue',arrows=True)
        plt.title("Gradient Reflector tracked graph operation")
        plt.show()