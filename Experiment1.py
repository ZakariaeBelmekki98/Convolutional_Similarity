"""
    Experiment from section 4.1


"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def CS(v1, v2):
    return (torch.dot(v1, v2)/torch.norm(v1)/torch.norm(v2))

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
LR = 0.1


if __name__ == "__main__":
    total_f_cs_reduction = 0.0
    total_k_cs_reduction = 0.0
     

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
            loss = CS(K1, K2)**2
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
        ax1.plot(loss_list, label="Kernel Cosine Similarity")
        ax1.set_ylim([-0.05, 1.05])
        ax2.plot(feature_cs_list, label="Feature Map Cosine Similarity")
        ax2.set_ylim([-0.05, 1.05])
        ax1.get_shared_x_axes().join(ax1, ax2)
        ax1.set_xticklabels([])

        final_f_cs = CS(F1, F2)
        final_k_cs = CS(K1, K2)
        
        f_reduction =  (abs(init_f_cs) - abs(final_f_cs))/abs(init_f_cs)
        k_reduction = (abs(init_k_cs) - abs(final_k_cs))/abs(init_k_cs)
        
        total_f_cs_reduction += f_reduction
        total_k_cs_reduction += k_reduction
    print("Kernels CS reduction: {:.4f}".format(total_k_cs_reduction/REPS))
    print("Feature maps CS reduction: {:.4f} ".format(total_f_cs_reduction/REPS))
    
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.show()
