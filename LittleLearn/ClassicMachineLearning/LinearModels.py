import numpy as np 
from typing import Literal
import traceback

class LinearRegression :
    def __init__ (self,learning_rate = 0.01,
                  optimizer = None ,
                  loss = None ) :
        self.optimizer = optimizer
        self.learning_rate = learning_rate 
        self.__record_loss = list()
        self.Weight = None 
        self.bias = None 
        self.loss = loss 
        self.optimizer = optimizer
    
    def build_Models (self,features) : 
        self.Weight = np.random.normal(size=(features,1)) * (1 / np.sqrt(features))
        self.bias = np.random.normal() 

    def fit(self,X,Y,Verbose : Literal[0,1] = 1,epochs=100) : 
        if self.Weight is None or self.bias is None :
            self.build_Models(X.shape[1]) 
        for epoch in range(epochs) :
            if len(X.shape) != 2 :
                print(f"Warning :: this X shape is = {X.shape}.X input must 2 dimentional do X.rehape(-1,1) before train")
                break
            y_pred = np.dot(X,self.Weight) + self.bias
            if self.loss is None :
                loss = np.mean(np.power((y_pred - Y),2))
            else :
                loss = self.loss(Y,y_pred)
            if self.optimizer is None :
                if len(Y.shape) != 2 :
                    print(f"Warning : Y shape is : {Y.shape} is Not Compatible You must do reshape to Y.reshape(-1,1)")
                    break
                else :
                    gradient_w = (-2/len(Y)) * np.dot(X.T,(Y - y_pred))
                    gradient_b = (-2/len(Y)) * np.sum(Y - y_pred)
                    self.Weight -= self.learning_rate * gradient_w
                    self.bias -= self.learning_rate * gradient_b
            elif self.optimizers is not None :
                try :
                    gradient_w = (1/len(Y)) * np.dot(X.T,(y_pred - Y))
                    gradient_b = (1/len(Y)) * np.sum(y_pred-Y)
                    grad = self.optimizers(self.Weight,self.bias,gradient_w,gradient_b,epoch)
                    self.Weight = grad['weight']
                    self.bias = grad['bias']
                except :
                    gradient_w = (1/len(Y)) * np.dot(X.T,(y_pred - Y))
                    gradient_b = (1/len(Y)) * np.sum(y_pred-Y)
                    grad = self.optimizers(self.Weight,self.bias,gradient_w,gradient_b)
                    self.Weight = grad['weight']
                    self.bias = grad['bias']
            if Verbose != 1 or Verbose != 0:
                try :
                    raise ValueError("Verbose Values is not Valid")
                except Exception as e :   
                    e.add_note("Error : Verbose just choice [0 / 1] and you input {Verbose}".format(Verbose))
                    traceback.print_exception(type(e),e,e.__traceback__)
            if Verbose ==  1 :
                print(f"epoch : {epoch} || Loss : {loss:.6f}")
            self.__record_loss.append(loss)
            
    @property 
    def get_loss_record (self) :
        try :
            if len(self.__record_loss) == 0:
                raise ValueError("Model still not trained")
            return np.array(self.__record_loss)
        except Exception as e :
            e.add_note("You must Training model first")
            traceback.print_exception(type(e),e,e.__traceback__)

    
    def __call__ (self,X) :
        try :
            if self.Weight is None or self.bias in None :
                ValueError(f"None of Weight and bias can't do prediction")
            return np.dot(X,self.Weight) + self.bias
            
        except Exception as e :
            if self.Weight is None or self.bias is None : 
                e.add_note("Error : You must do Model.build_Models(features) at your model")
                e.add_note(f"Detail : Weight : {self.Weight} bias : {self.bias}")
                traceback.print_exception(type(e),e,e.__traceback__)
                raise

class LogisticRegression :
    def __init__ (self,learning_rate =0.001,optimizer=None,epsilon=1e-5) :
        self.learning_rate = learning_rate
        self.optimizers = optimizer 
        self.Weight = None 
        self.bias = None 
        self.__record_loss = list()
        self.__record_accuracy = list()
        self.epsilon = epsilon
    
    def build_Models (self,features) :
        self.Weight = np.random.normal(size=(features,1)) * (1/np.sqrt(features))
        self.bias = np.random.normal()

    def fit(self,X,Y,epochs=100,verbose:Literal[0,1]=1) :
        if self.Weight is None or self.bias is None :
            self.build_Models(X.shape[1])
        for epoch in range(epochs) :
            if len(Y.shape) != 2 or len(X.shape) < 2:
                print(f"Y or Xmust be 2 dims do Y.rehape(-1,1) or X.reshape(-1,1) before train models")
                break
            elif len(X.shape) >2 :
                print("The models just can fit on 2 dims data")
                break 
            scores = np.dot(X,self.Weight) + self.bias 
            y_pred = 1 / (1 + np.exp(-scores))
            loss = (-1/len(Y)) * np.sum(Y * (np.log(y_pred + self.epsilon)) + (1 - Y) * np.log(1-y_pred + self.epsilon))
            accuracy = len(Y==y_pred)/len(Y)
            if self.optimizers is None :
                gradient_w = (1/len(Y)) * np.dot(X.T,(y_pred - Y))
                gradient_b = (1/len(Y)) * np.sum(y_pred-Y)
                self.Weight -= self.learning_rate * gradient_w
                self.bias -= self.learning_rate * gradient_b
            elif self.optimizers is not None :
                try :
                    gradient_w = (1/len(Y)) * np.dot(X.T,(y_pred - Y))
                    gradient_b = (1/len(Y)) * np.sum(y_pred-Y)
                    grad = self.optimizers(self.Weight,self.bias,gradient_w,gradient_b,epoch)
                    self.Weight = grad['weight']
                    self.bias = grad['bias']
                except :
                    gradient_w = (1/len(Y)) * np.dot(X.T,(y_pred - Y))
                    gradient_b = (1/len(Y)) * np.sum(y_pred-Y)
                    grad = self.optimizers(self.Weight,self.bias,gradient_w,gradient_b)
                    self.Weight = grad['weight']
                    self.bias = grad['bias']
            if verbose !=1 or verbose != 0:
                try:
                    raise ValueError("verbose values is not valid")
                except Exception as e :
                    e.add_note("Use 1 or 0 for show detail training")
                    traceback.print_exception(type(e),e,e.__traceback__)
                    raise
            if verbose == 1 :
                print(f"epoch : {epoch} || Loss : {loss:.6f} || Accuracy : {accuracy:.6f}")
            self.__record_loss.append(loss)
            self.__record_accuracy(accuracy)
    
    def __call__(self,X):
        try: 
            if self.Weight is None or self.bias is None :
                raise ValueError("Model can't do predicting if the Weight and bias is None")
            score = np.dot(X,self.Weight) + self.bias 
            return 1 / (1 + np.exp(-score))
        except Exception as e :
            e.add_note(f"error : Weight is {self.Weight} and bias is {self.bias}")
            e.add_note("use Model.build_Models(Feature) before do prediction")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise
    
    @property
    def get_loss_record (self) :
        try :
            if len(self.__record_loss) == 0:
                raise ValueError("Model still not trained")
            return np.array(self.__record_loss)
        except Exception as e :
            e.add_note("You must Training model first")
            traceback.print_exception(type(e),e,e.__traceback__)
            
    @property
    def get_accuracy_record (self) :
        try :
            if len(self.__record_loss) == 0:
                raise ValueError("Model still not trained")
            return np.array(self.__record_accuracy)
        except Exception as e :
            e.add_note("You must Training model first")
            traceback.print_exception(type(e),e,e.__traceback__)
            