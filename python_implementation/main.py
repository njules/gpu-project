import os
import ssl
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from eval import plot_losses, show_batch_images
from LeNet5 import LeNet5, load_model_npy, save_model_npy, save_model_txt
from train import train_epochs


ssl._create_default_https_context = ssl._create_unverified_context  # for downloading dataset
np.set_printoptions(threshold=sys.maxsize)  # for saving weights in txt file


DEVICE = 'cuda'  # 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.01
BATCH_SIZE = 32
N_EPOCHS = 15
CLASSES = (
  'plane', 'car', 'bird', 'cat', 'deer',
  'dog', 'frog', 'horse', 'ship', 'truck'
)


def main():

  # define transforms
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  # download and create datasets
  train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
  )
  val_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
  )

  # define the data loaders
  train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
  )
  val_loader = DataLoader(
    dataset=val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False
  )

  # initialize model and optimizer
  model = LeNet5(len(CLASSES))
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

  ################# train model ###########################
  # model, _, (train_losses, val_losses) = train_epochs(model, criterion, optimizer, train_loader, val_loader, N_EPOCHS, DEVICE)
  # plot_losses(train_losses, val_losses)
  # save_model_npy(model, 'weights/npy')
  model = load_model_npy(model, 'weights/npy')
  save_model_txt(model, 'weights/txt')

  ################# sample images #########################
  # data_iter = iter(train_loader)
  # inputs, labels = next(data_iter)
  # _, outputs = model(inputs)
  # preds = outputs.max(dim=1).indices

  ################# show sample images ####################
  # show_batch_images(inputs)
  # print(' '.join('%5s' % CLASSES[preds[j]] for j in range(labels.shape[0])))
  # print(' '.join('%5s' % CLASSES[labels[j]] for j in range(labels.shape[0])))

  ################# select sample image ###################
  # idx = (preds == labels).max(dim=0).indices
  # image = inputs[idx]
  # image_unprocessed = image / 2 + 0.5
  # label = CLASSES[labels[idx]]

  # save/load sample image
  # file_path = os.path.join('test_images', label)
  # torchvision.utils.save_image(image, f'{file_path}_processed.png')
  # torchvision.utils.save_image(image_unprocessed, f'{file_path}_raw.png')
  # np.save(f'{file_path}.npy', image.numpy())
  # with open(f'{file_path}.txt', 'w') as save_file:
  #   save_file.write(np.array2string(image.numpy(), separator=','))

  ################# measure time ##########################
  image = torch.Tensor(np.load('test_images/frog.npy'))
  image = image.unsqueeze(dim=0)
  image = image.to(DEVICE)
  model = model.to(DEVICE)
  with profile(record_shapes=True, profile_memory=True, with_flops=True) as prof:
    _, output = model(image)
  prediction = CLASSES[output.max(dim=1).indices]
  print(f"Profiling single forward pass of {prediction} image. ")
  print(prof.key_averages())
  prof.export_chrome_trace("python_implementation/trace.json")




if __name__ == '__main__':
  main()