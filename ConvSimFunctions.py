import torch
import convsim

class ConvSimFunctionCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        inner_prods = convsim.inner_prods(input)
        output = torch.tensor(convsim.conv_sim(inner_prods, input.shape[0], input.shape[1], input.shape[2] * input.shape[3]))
        ctx.save_for_backward(inner_prods, input)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        outputs = grad_out * convsim.conv_sim_grad(*ctx.saved_tensors)
        return outputs


class ConvSim2DFunctionCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        convs = convsim.convs_2D(input)
        output = torch.tensor(convsim.conv_sim_2D(convs, input.shape[0], input.shape[1], input.shape[2]))
        ctx.save_for_backward(convs, input)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        outputs = grad_out * convsim.conv_sim_2D_grad(*ctx.saved_tensors)
        return outputs


ConvSim = ConvSimFunctionCUDA.apply
ConvSim2D = ConvSim2DFunctionCUDA.apply


def ConvSimLoss(model):
    loss = 0.0
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, torch.nn.Conv2d):
            loss += ConvSim(layer.weight)
    return loss

def ConvSim2DLoss(model, verbose=False):
    loss = 0.0
    cnt = 0
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, torch.nn.Conv2d):
            loss += ConvSim2D(layer.weight)
            cnt += 1
    if verbose:
        print("Number of convolutions: {}".format(cnt))
    return loss

