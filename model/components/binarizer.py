from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F


class SignFunction(Function):
    def __init__(self):
        super(SignFunction,self).__init__()
    @staticmethod
    def forward(ctx,input, is_train=True):
        if is_train:
            prob = input.new(input.size()).uniform_()
            x = input.clone()
            x[(1 - input) / 2 <= prob] = 1
            x[(1 - input) / 2 > prob] = -1
            return (x+1)//2
        else:
            return (input.sign()+1)//2
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
        
class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer,self).__init__()
        self.sign = Sign()
    def forward(self,x, is_train):
        x =  F.tanh(x)
        return SignFunction.apply(x, is_train)

class NaiveBinarizer(nn.Module):
    def __init__(self):
        super(NaiveBinarizer,self).__init__()
    def forward(self,x, is_train):
        return (x.sign()+1)//2