import LittleLearn as ll

def relu (x) :
    if not isinstance(x,ll.GradientReflector) :
        x = ll.GradientReflector(x)
    return x.relu()

def leaky_relu (x,alpha=1e-6) :
    if not isinstance(x,ll.GradientReflector) :
        x = ll.GradientReflector(x)
    return x.leaky_relu(alpha=alpha)

def swish(x,beta=1.0) :
    if not isinstance(x,ll.GradientReflector) :
        x = ll.GradientReflector(x)
    return x.swish(Beta=beta)

def gelu (x) :
    if not isinstance(x,ll.GradientReflector) :
        x = ll.GradientReflector(x)
    return x.gelu()
    
def softmax(x,y_label=None,axis=None,keepdims=False,epsilon=1e-6) :
    if not isinstance(x,ll.GradientReflector) :
        x = ll.GradientReflector(x)
    return x.softmax(y_label=y_label,axis=axis,keepdims=keepdims,epsilon=epsilon)

def sigmoid (x) :
    if not isinstance(x,ll.GradientReflector) :
        x = ll.GradientReflector(x)
    return x.sigmoid()

def linear (x) :
    if not isinstance(x,ll.GradientReflector) :
        x = ll.GradientReflector(x)
    return x.linear()