import torch
import torch.nn as nn

class MLPNet(nn.Module):

    def __init__(self,input_dim = 5):
        super(MLPNet, self).__init__()
        # kernel
        self.fc1 = nn.Linear(64*input_dim, 1024, dtype=torch.float64)
        self.fc2 = nn.Linear(1024, 2048, dtype=torch.float64)
        self.fc3 = nn.Linear(2048, 64, dtype=torch.float64)
        #self.fc4 = nn.Linear(64, 64, dtype=torch.float64)
        
        # Set whether to readout activation
        self.readout = False

    def forward(self, x):
        xno = nn.functional.normalize(x)
        l1 = torch.relu(self.fc1(xno))
        l2 = torch.relu(self.fc2(l1))
        #l3 = torch.relu(self.fc3(l2))
        y = torch.relu(self.fc3(l2))
        
        if self.readout:
            return {'y': y}
        else:
            return y   
        
class MLPNet2(nn.Module):

    def __init__(self,input_dim = 5):
        super(MLPNet2, self).__init__()
        # kernel
        self.fc1 = nn.Linear(64*input_dim, 1024, dtype=torch.float64)
        self.fc2 = nn.Linear(1024, 2048, dtype=torch.float64)
        self.fc3 = nn.Linear(2048, 512, dtype=torch.float64)
        self.fc4 = nn.Linear(512, 64, dtype=torch.float64)
        
        # Set whether to readout activation
        self.readout = False

    def forward(self, x):
        l1 = torch.relu(self.fc1(x))
        l2 = torch.relu(self.fc2(l1))
        l3 = torch.relu(self.fc3(l2))
        y = torch.relu(self.fc4(l3))
        
        if self.readout:
            return {'y': y}
        else:
            return y   
        
class MLPNet3(nn.Module):

    def __init__(self,input_dim = 5):
        super(MLPNet3, self).__init__()
        # kernel
        self.fc1 = nn.Linear(64*input_dim, 512, dtype=torch.float64)
        self.fc2 = nn.Linear(512, 1024, dtype=torch.float64)
        self.fc3 = nn.Linear(1024, 256, dtype=torch.float64)
        self.fc4 = nn.Linear(256, 64, dtype=torch.float64)
        
        # Set whether to readout activation
        self.readout = False

    def forward(self, x):
        xno = nn.functional.normalize(x)
        l1 = torch.relu(self.fc1(xno))
        l2 = torch.relu(self.fc2(l1))
        l3 = torch.relu(self.fc3(l2))
        y = torch.relu(self.fc4(l3))
        
        if self.readout:
            return {'y': y}
        else:
            return y   