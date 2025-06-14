from . import activations
from . import DeepLearning 
from . import ClassicMachineLearning
from . import preprocessing 
from . import Loss
from . import optimizers
from .GradientReflector import GradientReflector

def convert_to_tensor(x) :
    return GradientReflector(x)

def matmul(matrix_a,matrix_b,transpose_a=False,transpose_b=False):
    try:
        if not isinstance (matrix_a,GradientReflector) :
            matrix_a = GradientReflector(matrix_a)
        return matrix_a.matmul(matrix_b,transpose_a=transpose_a,transpose_b=transpose_b)
    except :
        if matrix_a.shape == matrix_b.shape :
            raise RuntimeError(f"mismatch {matrix_a.shape} vs {matrix_b.shape} you have transpose a one of matriks")


def dot (matriks_a,matriks_b) :
    if not isinstance (matriks_a,GradientReflector):
        matriks_a = GradientReflector(matriks_a)
    return matriks_a.dot(matriks_b)

def sin(a) :
    if not isinstance(a,GradientReflector) :
        a = GradientReflector(a)
    return a.sin()

def cos (a) :
    if not isinstance(a,GradientReflector):
        a = GradientReflector(a) 
    return a.cos()

def reshape(matriks,shape=()):
    try :
        if not isinstance(matriks,GradientReflector):
            matriks = GradientReflector(matriks)
        return matriks.reshape(shape=shape)
    except :
        if len(shape) == 0:
            raise RuntimeError("you have to gift a new shape for this matriks")
        raise RuntimeError (f"can't reshape at {shape}")

def log(x) :
    if not isinstance(x,GradientReflector) :
        x = GradientReflector(x)
    return x.log()

def exp(x) :
    if not isinstance(x,GradientReflector) :
        x = GradientReflector(x)
    return x.exp()

def sum (x,axis=None,keepdims=False) :
    if not isinstance(x,GradientReflector) :
        x = GradientReflector(x)
    return x.sum(axis=axis,keepdims=keepdims)

def tan(x) :
    if not isinstance(x,GradientReflector) :
        x = GradientReflector(x)
    return x.tan()

def clip(vector_or_matriks,min_vals,max_vals) :
    if not isinstance(vector_or_matriks,GradientReflector) :
        vector_or_matriks = GradientReflector(vector_or_matriks)
    return vector_or_matriks.clip(min_vals=min_vals,max_vals=max_vals)

def pow (tensor_or_scallar,pow_values) :
    if not isinstance(tensor_or_scallar,GradientReflector) :
        tensor_or_scallar = GradientReflector(tensor_or_scallar)
    return tensor_or_scallar.pow(power_values=pow_values)
