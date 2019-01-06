#Author Aishwarya Budhkar

#The data has been decorrelated by rotating it on to the Eigenvectors of the covariance matrix of the translated data. 
#Each dimension is then normalized to have a variance of 1 by dividing the rotated data by the square root of eigenvalues.

import torch
import numpy as np
import matplotlib.pyplot as plt
data = torch.load('assign0_data.py')

#Function to plot data
def plot(X,title):
    plt.scatter(X[:,0], X[:,1])
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#Plot original data
plot(data,'Original data')

#subtract mean from each data point to get zero centered data
normed = data - torch.mean(data,0)

#Plot data with zero mean
plot(normed, 'Data centered at origin')
#Please close the plot to check further plots

#function to calculate covarience
def calculateCovariance(X):
    cov1 = torch.mm(X.t(),X)/(float(X.shape[0])-1)
    return cov1

#print covariance matrix
print("Covariance matrix after Zero centering data\n")
print(calculateCovariance(normed))

#function to decorrelate data
def decorrelate_data(X):
    cov1 = torch.mm(X.t(),X)/(float(X.shape[0])-1)
    U,eigValues,eigVectors = torch.svd(cov1)
    decorrelate_out = torch.mm(X,eigVectors)
    return decorrelate_out,eigValues

#function to whiten data
def whiten_data(X):
    decorrelate_out,eigValues=decorrelate_data(normed)
    whitened_out = decorrelate_out / torch.sqrt(eigValues)
    return whitened_out

whitened_out = whiten_data(normed)
print("\nCovariance matrix of whitened data")
print(calculateCovariance(whitened_out))

#plot whitened data
plot(whitened_out,'Whitened data')

#  Dimensions are statistically dependent in higher dimensions
