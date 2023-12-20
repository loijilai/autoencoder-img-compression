import torch
# A Straight-Through Estimator (STE) implementation of the stochastic binarization
# The quantization is modified from https://github.com/alexandru-dinu/cae/blob/master/src/models/cae_16x8x8_refl_pad_bin.py
# The class structure is modified from https://github.com/abskj/lossy-image-compression
class STEBinarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, is_train):
        if is_train:
            # During training, use the stochastic binarization
            input = torch.tanh(input) # normalize latent representation in range: [-1, 1]
            rand = torch.rand(input.shape).to(input.device) # used as a random threshold
            prob = (1 + input) / 2 # range: [0, 1], indicates how likely the quantized value will be 1
            eps = torch.zeros(input.shape).to(input.device) # used to add to latent representation to achieve binarization 
            eps[rand <= prob] = (1 - input)[rand <= prob]
            eps[rand > prob] = (-input - 1)[rand > prob]
            quantized = input + eps
            return quantized
        else:
            sign = torch.sign(input)
            return sign

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: Just pass the gradient
        return grad_output, None
    
# A wrapper for the STEBinarizer
class Binarizer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, is_train):
        return STEBinarizer.apply(x, is_train)

# binarizer = Binarizer()
# x = torch.tensor([-1, 0.5, 0.2, 0.8, 1], dtype=torch.float32, requires_grad=True)