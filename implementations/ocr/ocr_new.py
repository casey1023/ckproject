import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
data_dir = '../../data/ocr_data/'
image_size = 224
#load data
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                    transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.RandomResizedCrop(image_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
                                    )

val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                    transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
                                    )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 4, shuffle = True, num_workers=4)
num_classes = len(train_dataset.classes)
with open('log.txt','w') as f:
    print("-------",file=f,flush=True)
use_cuda = torch.cuda.is_available()

torch.cuda.set_device(0)
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor

#show dataset
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0)) 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()
    plt.pause(0.001)
images, labels = next(iter(train_loader))
out = torchvision.utils.make_grid(images)

imshow(out, title=[train_dataset.classes[x] for x in labels])

net = models.resnet18(pretrained=True)
net = net.cuda() if use_cuda else net

num_ftrs = net.fc.in_features

net.fc = nn.Linear(num_ftrs, 4803)

net.fc = net.fc.cuda() if use_cuda else net.fc

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.0001, momentum=0.9)

record = [] 

num_epochs = 50
net.train(True)
best_model = net
best_r = 0.0
def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    rights = rights.cpu() if rights.is_cuda else rights
    return rights, len(labels)
#run

for epoch in range(num_epochs):
    prev_time=time.time()
    #optimizer = exp_lr_scheduler(optimizer, epoch)
    train_rights = []
    train_losses = []
    for batch_idx, (data, target) in enumerate(train_loader): 
        data, target = data.clone().detach().requires_grad_(True), target.clone().detach()
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = rightness(output, target)
        train_rights.append(right)
        loss = loss.cpu() if use_cuda else loss
        train_losses.append(loss.data.numpy())

    train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))

    net.eval()
    test_loss = 0
    correct = 0
    vals = []
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = data.clone().detach().requires_grad_(False), target.clone().detach()
        output = net(data)
        val = rightness(output, target)
        vals.append(val)

    val_r = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
    val_ratio = 1.0*val_r[0].numpy()/val_r[1]
    
    if val_ratio > best_r:
        best_r = val_ratio
        best_model = copy.deepcopy(net)
    with open('log.txt','a') as f:
        now_time=time.time()
        print('epoch: {} \tLoss: {:.6f}\ttrain accuracy: {:.2f}%,test accuracy: {:.2f}%,time: {} : {}'.format(
        epoch, np.mean(train_losses), 100. * train_r[0].numpy() / train_r[1], 100. * val_r[0].numpy()/val_r[1],(now_time-prev_time)//60,(now_time-prev_time)%60),file=f,flush=True)       
    record.append([np.mean(train_losses), train_r[0].numpy() / train_r[1], val_r[0].numpy()/val_r[1]])

x = [x[0] for x in record]
y = [1 - x[1] for x in record]
z = [1 - x[2] for x in record]
#plt.plot(x)
plt.figure(figsize = (10, 7))
plt.plot(y)
plt.plot(z)
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.savefig('epoch-err_rate.png')
def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure(figsize=(15,10))

    for i, data in enumerate(val_loader):
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        preds = preds.cpu().numpy() if use_cuda else preds.numpy()
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot( 2,num_images//2, images_so_far)
            ax.axis('off')
            
            ax.set_title('predicted: {}'.format(val_dataset.classes[preds[j]]))
            imshow(data[0][j])

            if images_so_far == num_images:
                return
visualize_model(net)

plt.ioff()