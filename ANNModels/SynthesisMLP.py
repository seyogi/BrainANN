import torch
import torch.nn as nn

class SynthesisMLP(nn.Module):
    def __init__(self, mlp1, mlp2):
        super().__init__()

        # Continuous time RNN
        self.mlp1 = mlp1
        self.mlp2 = mlp2

    def forward(self, x):
        out = torch.zeros(1,64)
        out1 = self.mlp1(x)
        out2 = self.mlp2(x)
        for i in range(len(out1[0])):
            if out1[0][i] > 0 and out2[0][i] > 0:
                out[0][i] = (out1[0][i]+out2[0][i])/2
            elif out1[0][i] > 0 and out2[0][i] == 0:
                out[0][i] = out1[0][i]
            elif out1[0][i] == 0 and out2[0][i] > 0:
                out[0][i] = out2[0][i]
        return out
