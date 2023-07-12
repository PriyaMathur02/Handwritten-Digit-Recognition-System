import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import RandomRotation, RandomAffine
import PIL.ImageOps
import requests
from PIL import Image

def custom_collate(batch):
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Convert PIL.Image.Image objects to tensors
    data = torch.stack(data)

    return [data, targets]

transform1 = transforms.Compose([transforms.ToTensor(),transforms.Resize((28,28)),transforms.Normalize((0.5,),(0.5,))])
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomApply([RandomRotation(30)], p=0.5),
    transforms.RandomApply([RandomAffine(0, translate=(0.1, 0.1))], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform1)
valid_dataset=datasets.MNIST(root='./data', train=False, download=True, transform=transform1)
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100,shuffle=True,collate_fn=custom_collate)
valid_loader=torch.utils.data.DataLoader(dataset=valid_dataset,batch_size=100,shuffle=False,collate_fn=custom_collate)

#convert tensor to numpy
#clone means to copy the tensor the detach it to convert it to numpy
"""  transposition effectively rearranges the dimensions of the image array to
match the order commonly used in NumPy arrays for representing images, which is (height, width, channels).
In the original image tensor, the dimensions are typically ordered as (batch_size, channels, height, width).
The number 0 corresponds to the first dimension (channels), 1 corresponds to the second dimension (height),
and 2 corresponds to the third dimension (width).In the line image.transpose(1, 2, 0), the numbers 1, 2, 
and 0 specify the new order of dimensions in the transposed array."""

"""This line clips the pixel values of the image array to the range [0, 1]. Any values below 0 are set to 0, and any
values above 1 are set to 1. This ensures that the pixel values remain within a valid range for image representation."""


def convert(tensor): 
    image=tensor.clone().detach().numpy()
    image=image.transpose(1,2,0)
    print(image.shape) #prints the shape of image array(height,width,channel)
    image=image*(np.array((0.5,0.5,0.5))+np.array((0.5,0.5,0.5))) #image normalization 
    image=image.clip(0,1)  
    return image  

dataiter = iter(train_loader)
images, labels = next(dataiter)
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1)
    plt.imshow(convert(images[idx]))
    ax.set_title(str(labels[idx]))  # convert to str
    
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    
class Classification(nn.Module):

    def __init__(self, inputlayer, hiddenlayer1, hiddenlayer2, outputlayer):
        super().__init__()
        self.linear1 = nn.Linear(inputlayer, hiddenlayer1)
        self.linear2 = nn.Linear(hiddenlayer1, hiddenlayer2)
        self.linear3 = nn.Linear(hiddenlayer2, outputlayer)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Defining the parameters   
model = Classification(784, 125, 65, 10)  

# Loss function
criterion = nn.CrossEntropyLoss()  

# Optimization algorithm
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)   #learning rate=0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 15  
loss_history = []  
correct_history = []  
val_loss_history = []  
val_correct_history = []

for e in range(epochs):  
    loss=0.0  
    correct=0.0  
    val_loss=0.0  
    val_correct=0.0  

    
    for input,labels in train_loader:  
        inputs=input.view(input.shape[0],-1)  
        labels=torch.tensor(labels) #update
        outputs=model(inputs)  
        loss1=criterion(outputs,labels)  
        optimizer.zero_grad() 
        
        loss1.backward() 
        
        optimizer.step()  
        _,preds=torch.max(outputs,1)  
        loss+=loss1.item()  
        correct+=torch.sum(preds==labels.data)  
        
    else:  
        with torch.no_grad():  
            for val_input,val_labels in valid_loader:  
                val_inputs=val_input.view(val_input.shape[0],-1)
                val_labels=torch.tensor(val_labels)
                val_outputs=model(val_inputs)  
                val_loss1=criterion(val_outputs,val_labels)   
                _,val_preds=torch.max(val_outputs,1)  
                val_loss+=val_loss1.item()  
                val_correct+=torch.sum(val_preds==val_labels.data)  
                
                
        epoch_loss=loss/len(train_loader.dataset)  
        epoch_acc=correct.float()/len(train_dataset)  
        loss_history.append(epoch_loss)  
        correct_history.append(epoch_acc)  
          
        val_epoch_loss=val_loss/len(valid_loader.dataset)  
        val_epoch_acc=val_correct.float()/len(valid_dataset)  
        val_loss_history.append(val_epoch_loss)  
        val_correct_history.append(val_epoch_acc)  
        print('training_loss:{:.4f},{:.4f}'.format(epoch_loss,epoch_acc.item()))  
        print('validation_loss:{:.4f},{:.4f}'.format(val_epoch_loss,val_epoch_acc.item()))  
