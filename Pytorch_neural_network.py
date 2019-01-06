# Author Aishwarya Budhkar

# Neural network with one hidden layer for approximating the cosine function

import torch
import matplotlib.pyplot as plt
import numpy as np

# Generate function y = cos(x) over given interval
input = torch.unsqueeze(torch.linspace(-1*np.pi, np.pi,int(2*np.pi/0.01) ), dim=1)  # x data (tensor), shape=(100, 1)
output = torch.cos(input)
plt.title("Expected curve")
plt.scatter(input,output)
plt.show()

class Net(torch.nn.Module):

    def __init__(self, N_IN, N_HID,N_OUT):

        super(Net, self).__init__()
        self.hid = torch.nn.Linear(N_IN, N_HID)
        self.out = torch.nn.Linear(N_HID, N_OUT)

    def forward(self, x):
        
        #tanh activation for hidden layer
        x = torch.tanh(self.hid(x))
        x = self.out(x)
        return x

# 10 neurons in hidden layer
net = Net(N_IN=1, N_HID=10, N_OUT=1)

# Mean square error
loss_squared_error_func = torch.nn.MSELoss()

# Adam optimizer with learning rate 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Number of epochs 3000
for epoch in range(3000):
    predicted_out = net(input)

    # predicted  over expected loss
    loss = loss_squared_error_func(predicted_out, output)
    optimizer.zero_grad()

    # backprop
    loss.backward()
    optimizer.step()
plt.scatter(input.data.numpy(), output.data.numpy())
plt.title("Predicted output")
plt.plot(input.data.numpy(), predicted_out.data.numpy(), 'r-', lw=5)
plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
plt.show()
