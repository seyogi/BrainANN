import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch

import neurogym as ngym
import pickle

from datetime import datetime

import ANNModels.RNNnet as RNNnet

def RNNTrain(net,dataset,optimizer,criterion,output_size,training_num = 1000):
    running_loss = 0
    running_acc = 0
    for i in range(training_num):
        inputs, labels_np = dataset()
        labels_np = labels_np.flatten()
        inputs = torch.from_numpy(inputs).type(torch.float)
        labels = torch.from_numpy(labels_np).type(torch.long)
        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(inputs)
        output = output.view(-1, output_size)
        if i<1:
            print(output, labels)
            print(criterion(output, labels))
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()    # Does the update

        running_loss += loss.item()
        # Compute performance
        output_np = np.argmax(output.detach().numpy(), axis=-1)
        ind = labels_np > 0  # Only analyze time points when target is not fixation
        running_acc += np.mean(labels_np[ind] == output_np[ind])
        if i % 100 == 99:
            running_loss /= 100
            running_acc /= 100
            print('Step {}, Loss {:0.4f}, Acc {:0.3f}'.format(i+1, running_loss, running_acc))
            running_loss = 0
            running_acc = 0
    return net

def main():
    dataset_name = ".\CreateDelayTask\_2024_09_12_14_53_35"
    with open(dataset_name +'.pickle', mode='br') as file:
        dataset = pickle.load(file)

        env = dataset.env
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n
        print("input_size :",input_size)
        print("output_size:",output_size)

        hidden_size = 64

        net = RNNnet.RNNNet(input_size=input_size, hidden_size=hidden_size,
                    output_size=output_size, dt=env.dt)
        print(net)

        # Use Adam optimizer
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        net = RNNTrain(net,dataset,optimizer,criterion,output_size)
        
        exportFlag = input('Export this model (y/n) :')
        if exportFlag == 'y':
            date = datetime.now().strftime("%m%d")
            filename = 'RNN_'+date+"_" + input('Filename:'+'RNN_'+date+"_")
            with open( ".\\ANNModels\\RNN\\" + filename + '.pickle', mode='wb') as fo:
                pickle.dump(net, fo)

if __name__ == "__main__":
    main()