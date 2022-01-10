# This code is used to train MNIST and CIFAR10 datasets for feature inversion attacks

import time
import math
import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from f_utils import setLearningRate, evalTest
from network import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def train(DATASET, network, NEpochs, imageWidth,
        imageHeight, imageSize, NChannels, NClasses,
        BatchSize, learningRate,NDecreaseLR, eps,
        AMSGrad, model_dir, model_name, gpu = True):

    print("DATASET: ", DATASET)

    if DATASET == 'MNIST':

        mu = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())

        tsf = {
            'train': transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))##
            ]),

            'test': transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))##
            ])
            }

        trainset = torchvision.datasets.MNIST(root='./data/MNIST', train=True,
                                        download=True, transform = tsf['train'])

        testset = torchvision.datasets.MNIST(root='./data/MNIST', train=False,
                                       download=True, transform = tsf['test'])

    elif DATASET == 'CIFAR10':

        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
        tsf = {
            'train': transforms.Compose(
            [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees = 10, translate = [0.1, 0.1], scale = [0.9, 1.1]),
            transforms.ToTensor(),
            Normalize
            ]),
            'test': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize
            ])
        }
        trainset = torchvision.datasets.CIFAR10(root='./data', train = True,
                                        download=True, transform = tsf['train'])##
        testset = torchvision.datasets.CIFAR10(root='./data', train = False,
                                       download=True, transform = tsf['test'])##

    netDict = {
        'MNISTNet': MNISTNet,
        'CIFAR10CNN': CIFAR10CNN
    }

    if network in netDict:
        net = netDict[network](NChannels)
    else:
        print("Network not found")
        exit(1)

    print(net)
    print("len(trainset) ", len(trainset))
    print("len(testset) ", len(testset))
    x_train, y_train = trainset[0]##
    x_test, y_test = testset[0]##

    print("x_train.shape ", x_train.shape)
    print("x_test.shape ", x_test.shape)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = BatchSize,
                                      shuffle = True, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                      shuffle = False, num_workers = 1)
    trainIter = iter(trainloader)
    testIter = iter(testloader)

    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    if gpu:
        net.cuda()
        criterion.cuda()
        softmax.cuda()

    optimizer = optim.Adam(params = net.parameters(), lr = learningRate, eps = eps, amsgrad = AMSGrad)

    NBatch = int(len(trainset) / BatchSize)##
    cudnn.benchmark = True
    for epoch in range(NEpochs):
        lossTrain = 0.0
        accTrain = 0.0
        for i in range(NBatch):
            try:
                batchX, batchY = trainIter.next()
            except StopIteration:
                trainIter = iter(trainloader)
                batchX, batchY = trainIter.next()

            if gpu:
                batchX = batchX.cuda()
                batchY = batchY.cuda()

            optimizer.zero_grad()
            logits = net.forward(batchX)
            prob = softmax(logits)

            if DATASET == 'MNIST':
               logits=logits.reshape(32,10)##
            elif DATASET == 'CIFAR10':
               logits=logits.reshape(32,40)##

            loss = criterion(logits, batchY)
            loss.backward()
            optimizer.step()

            lossTrain += loss.cpu().detach().numpy() / NBatch
            if gpu:
                pred = np.argmax(prob.cpu().detach().numpy(), axis = 1)
                groundTruth = batchY.cpu().detach().numpy()
            else:
                pred = np.argmax(prob.detach().numpy(), axis = 1)
                groundTruth = batchY.detach().numpy()

            acc = np.mean(pred == groundTruth)
            accTrain += acc / NBatch

        if (epoch + 1) % NDecreaseLR == 0:
            learningRate = learningRate / 2.0
            setLearningRate(optimizer, learningRate)

        print("Epoch: ", epoch, "Loss: ", lossTrain, "Train accuracy: ", accTrain)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache() ##free extraneous memory
        accTest = evalTest(testloader, net, gpu = gpu)


    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(net, model_dir + model_name)
    print("Model saved")

    newNet = torch.load(model_dir + model_name)
    newNet.eval()
    accTest = evalTest(testloader, net, gpu = gpu)
    print("Model restore done")


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type = str, default ='MNIST')
        parser.add_argument('--network', type = str, default ='MNISTNet')
        parser.add_argument('--epochs', type = int, default = 200)
        parser.add_argument('--eps', type = float, default = 1e-3)
        parser.add_argument('--AMSGrad', type = bool, default = True)
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--learning_rate', type = float, default = 1e-3)
        parser.add_argument('--decrease_LR', type = int, default = 20)

        parser.add_argument('--nogpu', dest='gpu', action='store_false')
        parser.set_defaults(gpu=True)
        args = parser.parse_args()

        model_dir = "checkpoints/" + args.dataset + '/'
        model_name = "ckpt.pth"

        if args.dataset == 'MNIST':

            imageWidth = 28
            imageHeight = 28
            imageSize = imageWidth * imageHeight
            NChannels = 1
            NClasses = 10
            network = 'MNISTNet'##

        elif args.dataset == 'CIFAR10':

            imageWidth = 32
            imageHeight = 32
            imageSize = imageWidth * imageHeight
            NChannels = 3
            NClasses = 10
            network = 'CIFAR10CNN'

        else:
            print("No Dataset Found")
            exit(0)

        train(DATASET = args.dataset, network = network, NEpochs = args.epochs, imageWidth = imageWidth,
        imageHeight = imageHeight, imageSize = imageSize, NChannels = NChannels, NClasses = NClasses,
        BatchSize = args.batch_size, learningRate = args.learning_rate, NDecreaseLR = args.decrease_LR, eps = args.eps,
        AMSGrad = args.AMSGrad, model_dir = model_dir, model_name = model_name, gpu = args.gpu)

    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
