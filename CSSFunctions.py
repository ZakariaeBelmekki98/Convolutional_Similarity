import torch
import ZaiCUDA as zaic

class CSCS3DFunctionCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        inner_prods = zaic.inner_prods(input)
        output = torch.tensor(zaic.CSCS(inner_prods, input.shape[0], input.shape[1], input.shape[2] * input.shape[3]))
        ctx.save_for_backward(inner_prods, input)
        return output
    
    @staticmethod 
    def backward(ctx, grad_output):
        outputs =  grad_output * zaic._grad_CSCS(*ctx.saved_tensors)
        return outputs

class CSCSFunctionCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        inner_prods = zaic.inner_prods2(input)
        output = torch.tensor(zaic.CSCS2(inner_prods, input.shape[0], input.shape[1], input.shape[2] * input.shape[3]))
        ctx.save_for_backward(inner_prods, input)
        return output
    
    @staticmethod 
    def backward(ctx, grad_out):
        outputs = zaic._grad_CSCS2(*ctx.saved_tensors)
        return outputs
    
class CSS3FunctionCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        inner_prods = zaic.inner_prods3(input)
        output = torch.tensor(zaic.CSS3(inner_prods, input.shape[0], input.shape[1], input.shape[2] * input.shape[3], 0))
        ctx.save_for_backward(inner_prods, input)
        return output
    
    @staticmethod 
    def backward(ctx, grad_out):
        outputs = grad_out * zaic._grad_CSS3(*ctx.saved_tensors, 0)
        return outputs

class MCSS3FunctionCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        inner_prods = zaic.inner_prods3(input)
        output = torch.tensor(zaic.CSS3(inner_prods, input.shape[0], input.shape[1], input.shape[2] * input.shape[3], 1))
        ctx.save_for_backward(inner_prods, input)
        return output
    
    @staticmethod 
    def backward(ctx, grad_out):
        outputs = grad_out * zaic._grad_CSS3(*ctx.saved_tensors, 1)
        return outputs
class CSCS4FunctionCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        inner_prods = zaic.inner_prods4(input)
        output = torch.tensor(zaic.CSCS4(inner_prods, input.shape[0], input.shape[1], input.shape[2] * input.shape[3]))
        ctx.save_for_backward(inner_prods, input)
        return output
    
    @staticmethod 
    def backward(ctx, grad_out):
        outputs = grad_out * zaic._grad_CSCS4(*ctx.saved_tensors)
        return outputs

class NCSSFunctionCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        inner_prods = zaic.inner_prods5(input)
        output = torch.tensor(zaic.NCSS(inner_prods, input.shape[0], input.shape[1], input.shape[2] * input.shape[3], 0))
        ctx.save_for_backward(inner_prods, input)
        return output 
    
    @staticmethod
    def backward(ctx, grad_out):
        outputs = grad_out * zaic._grad_NCSS(*ctx.saved_tensors, 0)
        return outputs

class MNCSSFunctionCUDA(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, input):
        inner_prods = zaic.inner_prods5(input)
        output = torch.tensor(zaic.NCSS(inner_prods, input.shape[0], input.shape[1], input.shape[2] * input.shape[3], 1))
        ctx.save_for_backward(inner_prods, input)
        return output 
    
    @staticmethod
    def backward(ctx, grad_out):
        outputs = grad_out * zaic._grad_NCSS(*ctx.saved_tensors, 1)
        return outputs

def z_init(weight):
    M = weight.shape[0]
    C = weight.shape[1]
    N = weight.shape[2] * weight.shape[3]

    tmp = weight.view(M, C, N)
    for i in range(M):
        for c in range(C):
            for j in range(i):
                for k in range(N):
                    tmp[i, c] -= torch.dot(tmp[i, c], torch.roll(tmp[j, c], k))/torch.dot(tmp[j, c], tmp[j, c])*torch.roll(tmp[j, c], k)
        tmp[i, c] /= torch.norm(tmp[i, c])

    return tmp 

def CSSinit(model):
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, torch.nn.Conv2d):
            print(i)
            layer.weight =torch.nn.Parameter(zaic.z_init(layer.weight))
