"""
	Experiment from Section 4.3

"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def CS(v1, v2):
    return (torch.dot(v1, v2)/torch.norm(v1)/torch.norm(v2))

def ConvSim(v1, v2):
    N = v1.shape[0]
    result = 0.0
    for b in range(1-N, N):
        tmp = 0.0
        for j in range(N):
            if(j+b < N and j+b>=0):
                tmp += v1[j] * v2[j+b] 
        result += tmp*tmp
    return result

def convolve(X, K):
    M = X.shape[0]
    N = K.shape[0]

    result = torch.zeros((M-N+1), dtype=torch.float)
    for i in range(M-N+1):
        for j in range(N):
            if(i+j < M):
                result[i] += K[j] * X[i+j]
    return result

EPOCHS = 300
REPS = 1
N = 9
M = 64
LR = 0.01


# Experiments from section 4
if __name__ == "__main__":
    total_f_cs_reduction = 0.0
    total_k_cs_reduction = 0.0
    total_relu_cs_reduction = 0.0
    total_elu_cs_reduction = 0.0
    total_lrelu_cs_reduction = 0.0
     
    elu = torch.nn.ELU()
    relu = torch.nn.ReLU()
    lrelu = torch.nn.LeakyReLU()

    for i in range(REPS):
        K1 = torch.rand((N,), dtype=torch.float)
        K2 = torch.rand((N,), dtype=torch.float)
        X = torch.rand((M,), dtype=torch.float)
                
        F1 = convolve(X, K1)
        F2 = convolve(X, K2)

        init_f_cs = CS(F1, F2)
        init_k_cs = CS(K1, K2)

        optimizer = torch.optim.SGD([K1, K2], lr=LR) 
        K1.requires_grad_(True)
        K2.requires_grad_(True)
        
        loss_list = []
        feature_cs_list = []
        for j in range(EPOCHS):
            optimizer.zero_grad()
            loss = ConvSim(K1, K2)
            loss.backward()
            loss_list.append(loss.item())
            optimizer.step()

            with torch.no_grad():
                feature_cs_list.append(CS(convolve(X, K1), convolve(X, K2)))
        
        K1.requires_grad_(False)
        K2.requires_grad_(False)
        F1 = convolve(X, K1)
        F2 = convolve(X, K2)
        
        fig = plt.figure()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        ax1.plot(loss_list, label="Kernel Convolutional Similarity")
        ax2.plot(feature_cs_list, label="Feature Map Cosine Similarity")
        ax2.set_ylim([-0.05, 1.05])
        ax1.get_shared_x_axes().join(ax1, ax2)
        ax1.set_xticklabels([])

        final_f_cs = CS(F1, F2)
        final_k_cs = CS(K1, K2)
        final_relu_cs = CS(relu(F1), relu(F2))
        final_elu_cs = CS(elu(F1), elu(F2))
        final_lrelu_cs = CS(lrelu(F1), lrelu(F2))
        
        f_reduction =  (abs(init_f_cs) - abs(final_f_cs))/abs(init_f_cs)
        k_reduction = (abs(init_k_cs) - abs(final_k_cs))/abs(init_k_cs)
        relu_reduction = (abs(init_f_cs) - abs(final_relu_cs))/abs(init_f_cs)
        elu_reduction = (abs(init_f_cs) - abs(final_elu_cs))/abs(init_f_cs)
        lrelu_reduction = (abs(init_f_cs) - abs(final_lrelu_cs))/abs(init_f_cs)
        
        total_f_cs_reduction += f_reduction
        total_k_cs_reduction += k_reduction
        total_relu_cs_reduction += relu_reduction
        total_elu_cs_reduction += elu_reduction
        total_lrelu_cs_reduction += lrelu_reduction
    print("Kernels CS reduction: {:.4f}".format(total_k_cs_reduction/REPS))
    print("Feature maps CS reduction: {:.4f} ".format(total_f_cs_reduction/REPS))
    print("ReLU CS reduction: {:.4f}".format(total_relu_cs_reduction/REPS))
    print("ELU CS reduction: {:.4f}".format(total_elu_cs_reduction/REPS))
    print("LeakyReLU CS reduction: {:.4f}".format(total_lrelu_cs_reduction/REPS))
    
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.show()
