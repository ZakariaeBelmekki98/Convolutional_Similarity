"""

    This script contains the experiment for Section 4.4

"""

import torch
import numpy as np

def ConvSim(k1, k2):
    res = 0.0
    N = k1.shape[0]
    for i in range(1-N, N):
        tmp = 0.0
        for j in range(N):
            if(i+j < N and i+j>= 0):
                tmp += k1[j] * k2[j+i]
        res += tmp*tmp
    return res


EPOCHS = 10
if __name__ == "__main__":
    K1 = torch.rand((9,), dtype=torch.float)
    K2 = torch.rand((9,), dtype=torch.float)
    K3 = torch.rand((9,), dtype=torch.float)
    K4 = torch.rand((9,), dtype=torch.float)

    optimizer1 = torch.optim.SGD([K1, K2], lr=0.01)
    optimizer2 = torch.optim.SGD([K2, K3], lr=0.01)
    optimizer3 = torch.optim.SGD([K3, K4], lr=0.01)
    K = [K1, K2, K3, K4]
    # Initial Convolutional Similarity
    for i in range(4):
        for j in range(4):
            print(" {:.3f} ".format(ConvSim(K[i], K[j])), end="")

        print()
        
    print() 
    K1.requires_grad_(True)
    K2.requires_grad_(True)
    K3.requires_grad_(True)
    K4.requires_grad_(True)
    
    for i in range(EPOCHS):
        # K1 K2
        optimizer1.zero_grad()
        loss1 = ConvSim(K1, K2)
        loss1.backward()
        optimizer1.step()
        # K2 K3 
        optimizer2.zero_grad()
        loss2 = ConvSim(K2, K3)
        loss2.backward()
        optimizer2.step()

        # K3 K4 
        optimizer3.zero_grad()
        loss3 = ConvSim(K3, K4)
        loss3.backward()
        optimizer3.step()
    K1.requires_grad_(False)
    K2.requires_grad_(False)
    K3.requires_grad_(False)
    K4.requires_grad_(False)
    
    K = [K1, K2, K3, K4]
    # final Convolutional Similarity
    for i in range(4):
        for j in range(4):
            print(" {:.3f} ".format(ConvSim(K[i], K[j])), end="")

        print()
        

        
    
