import numpy as np 
from typing import Literal 
import traceback

class Sequential_Neural_Network :
    def __init__ (self,Layers = list(),Name='Stacking_Neural_Network') :
        self.layers = Layers
        self.__record_loss = list()
        self.__record_accuracy = list()
        self.Optimizers = None 
        self.Loss = None 
        self.Metrics = None 
        self.Parameters = None 
        self.__backpropogation_messege_Activations = "Derivative"
        self.__compile_state = False
        self.__Momentum_weight = None
        self.__Momentum_bias = None
        self.__Rmsprop_weight = None
        self.__Rmsprop_bias = None
        self.Length_model = 0
        self.__chache = None
        self.__Train = False
    
    def build_models (self,optimizers,Loss,Metrics) :
        self.Optimizers = optimizers
        self.Loss = Loss 
        self.Metrics = Metrics 
        self.__compile_state = True 
        self.Length_model = len(self.layers)
        

    def fit(self,X,Y,verbose : Literal[0,1]=1,epochs = 1000) :

        if self.Optimizers is None or self.Loss is None :
            try : 
                raise ValueError("The Models can't fit if Optimizers and Loss == None")
            except Exception as e :
                e.add_note("Messege : You Must import optimizers and use at model.compile(optimizers = your optimizers)")
                e.add_note("Messege : You Must import Loss and use at model.compile(Loss = your_loss)")
                traceback.print_exception(type(e),e,e.__traceback__)

        if self.__compile_state == False :
            try :
                raise ValueError("The model hasn't compile")
            except Exception  as e :
                e.add_note("Messege : do models.compile() before do fit at your model")
                traceback.print_exception(type(e),e,e.__traceback__)
                raise 
        if self.__Momentum_weight is None or self.__Momentum_bias is None :
            self.__Momentum_weight = list()
            self.__Momentum_bias = list()
            for i in range(len(self.layers)) :
                Momentum_weight = np.zeros_like(self.layers[i].weight)
                Momentum_bias = np.zeros_like(self.layers[i].bias)
                self.__Momentum_weight.append(Momentum_weight)
                self.__Momentum_bias.append(Momentum_bias)
        if self.__Rmsprop_weight is None or self.__Rmsprop_bias is None  :
            self.__Rmsprop_weight = list()
            self.__Rmsprop_bias = list()
            for i in range(len(self.layers)) :
                Rmsprop_weight = np.zeros_like(self.layers[i].weight)
                Rmprop_bias = np.zeros_like(self.layers[i].bias)
                self.__Rmsprop_weight.append(Rmsprop_weight)
                self.__Rmsprop_bias.append(Rmprop_bias)

        for epoch in range(epochs) :
            self.__Train = True
            self.__chache = list()
            y_pred = self.__call__(X)
            loss = self.Loss(Y, y_pred)
            self.__record_loss.append(loss)

            if self.Metrics:
                accuracy = self.Metrics(Y, y_pred)
                self.__record_accuracy.append(accuracy)

            if self.layers[-1].activation is not None :
                if self.layers[-1].activation.name != 'softmax' :
                    self.layers[-1].activation.Mode = self.__backpropogation_messege_Activations
            
            dL_dy = self.Loss.BackwardPass(Y, y_pred)
            if self.layers[-1].activation is None :
                grad_output = dL_dy * y_pred
            else :
                if self.layers[-1].activation.name != 'softmax' :
                    dy_dz = self.layers[-1].activation(y_pred)
                    grad_output = dL_dy * dy_dz
                elif self.layers[-1].activation.name == 'softmax' :
                    dy_dz = self.layers[-1].activation.backward(Y,y_pred)
                    grad_output = dL_dy * dy_dz

            a_prev = self.__chache[-2] if len(self.__chache) > 1 else X
            grad_w = np.dot(a_prev.T, grad_output)
            grad_b = np.sum(grad_output, axis=0)

            self.__apply_optimizer(-1, grad_w, grad_b, epoch)

            grad_ = grad_output
            for i in reversed(range(len(self.layers) - 1)):
                if self.layers[i].activation is not None :
                    self.layers[i].activation.Mode = self.__backpropogation_messege_Activations
                z = self.__chache[i]
                
                grad_input = np.dot(grad_, self.layers[i + 1].weight.T)
                if self.layers[i].activation is None : 
                    grad_ = grad_input * z
                else :
                    da_dz = self.layers[i].activation(z)
                    grad_ = grad_input * da_dz
                
                a_prev = self.__chache[i - 1] if i > 0 else X
                grad_w = np.dot(a_prev.T, grad_)
                grad_b = np.sum(grad_, axis=0)

                self.__apply_optimizer(i, grad_w, grad_b, epoch)

            for i in range(len(self.layers)) :
                if self.layers[i].activation is not None :
                    self.layers[i].activation.Mode = "Normal"
                    
            if verbose == 1 :
                if self.Metrics is not None :
                    print(f"epoch : {epoch} || accuracy : {accuracy:.6f} || loss : {loss:.6f}")
                else:
                    print(f"epoch : {epoch} || loss : {loss:.6f}")
            self.__chache = None 

            self.__Train = False 
            
    def __apply_optimizer(self, index, grad_w, grad_b, epoch):
        try:
            if self.Optimizers.name == 'momentum':
                self.Optimizers.change_component(
                    self.__Momentum_weight[index],
                    self.__Momentum_bias[index]
                )
            elif self.Optimizers.name == 'rmsprop':
                self.Optimizers.change_component(
                    self.__Rmsprop_weight[index],
                    self.__Rmsprop_bias[index]
                )
            else:
                self.Optimizers.change_component(
                    self.__Momentum_weight[index],
                    self.__Momentum_bias[index],
                    self.__Rmsprop_weight[index],
                    self.__Rmsprop_bias[index]
                )
        except:
            pass 

        try:
            result = self.Optimizers(
                self.layers[index].weight,
                self.layers[index].bias,
                grad_w,
                grad_b,
                epoch
            )
        except:
            result = self.Optimizers(
                self.layers[index].weight,
                self.layers[index].bias,
                grad_w,
                grad_b
            )
        
        self.layers[index].weight = result['weight']
        self.layers[index].bias = result['bias']

    @property 
    def get_loss_record (self) :
        try:
            if self.__record_loss is None :
                raise ValueError ("The record is None")
            return np.array(self.__record_loss)
        except Exception as e :
            e.add_note("You must Fit your Model First")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
    
    @property
    def get_record_accuracy (self) :
        try:
            if self.__record_accuracy is None :
                raise ValueError("The record is None")
        except Exception as e :
            e.add_note("You Must Fit your Model first")
            e.add_note("or maybe you not add Metrics for spesifik Model")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 
        
    def __call__ (self,x) :
        for layers in self.layers :
            x = layers(x)
            if self.__Train == True :
                self.__chache.append(x)
        return x 
    
    def model_details (self) :
        Layers_name = list()
        parameter_list = np.array([self.layers[i].parameter for i in range(len(self.layers))])
        for i in range(len(self.layers)) :
            Name = "layers " + str(i) + "Name : "+ self.layers[i].Name
            Layers_name.append(Name)
        for i in range(len(self.layers)) :
            print(Layers_name[i] + " " + "||", f"Parameters : {parameter_list[i]}")
        try :
            print(f"Parameters Total : {np.sum(parameter_list)}")
        except :
            print(f"Parameters Total : {None}")

