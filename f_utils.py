

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

def accuracy(predictions, labels):

    if not (predictions.shape == labels.shape):
        print("predictions.shape ", predictions.shape, "labels.shape ", labels.shape)
        raise AssertionError

    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu

def pseudoInverse(W):
    return np.linalg.pinv(W)


def getImgByClass(Itr, C = None):

    if C == None:
        return Itr.next()

    while (True):
        img, label = Itr.next()
        if label == C:
            break
    return img, label


def clip(data):
    data[data > 1.0] = 1.0
    data[data < 0.0] = 0.0
    return data

def preprocess(data):

    size = data.shape
    NChannels = size[-1]
    assert NChannels == 1 or NChannels == 3
    if NChannels == 1:
        mu = 0.5
        sigma = 0.5
    elif NChannels == 3:
        mu = [0.485, 0.456, 0.406]
        sigma = [0.229, 0.224, 0.225]
    data = (data - mu) / sigma

    assert data.shape == size
    return data


def deprocess(data):

    assert len(data.size()) == 4

    BatchSize = data.size()[0]
    assert BatchSize == 1

    NChannels = data.size()[1]
    if NChannels == 1:
        mu = torch.tensor([0.5,], dtype=torch.float32)##
        sigma = torch.tensor([0.5,], dtype=torch.float32)##
    elif NChannels == 3:
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    else:
        print("Unsupported image in deprocess()")
        exit(1)

    Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
    return clip(Unnormalize(data[0,:,:,:]).unsqueeze(0))


def evalTest(testloader, net, gpu = True):
    testIter = iter(testloader)
    acc = 0.0
    NBatch = 0
    for i, data in enumerate(testIter, 0):
        NBatch += 1
        batchX, batchY = data
        if gpu:
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        logits = net.forward(batchX)

        if gpu:
            pred = np.argmax(logits.cpu().detach().numpy(), axis = 1)
            groundTruth = batchY.cpu().detach().numpy()
        else:
            pred = np.argmax(logits.detach().numpy(), axis = 1)
            groundTruth = batchY.detach().numpy()
        acc += np.mean(pred == groundTruth)
    accTest = acc / NBatch
    print("Test accuracy: ", accTest) #, "NBatch: ", NBatch, "pred == groundTruth.shape", (pred == groundTruth).shape
    return accTest



def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)##
        if m.bias is not None:
            nn.init.normal_(m.bias.data)##
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)##
        if m.bias is not None:
            nn.init.normal_(m.bias.data)##
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)##
        if m.bias is not None:
            nn.init.normal_(m.bias.data)##
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)##
        if m.bias is not None:
            nn.init.normal_(m.bias.data)##
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)##
        if m.bias is not None:
            nn.init.normal_(m.bias.data)##
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)##
        if m.bias is not None:
            nn.init.normal_(m.bias.data)##
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

def setLearningRate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow(x[:,:,1:,:]-x[:,:,:h_x-1,:], 2).sum()
    w_tv = torch.pow(x[:,:,:,1:]-x[:,:,:,:w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def l2loss(x):
    return (x**2).mean()

def l1loss(x):
    return (torch.abs(x)).mean()

def getL1Stat(net, x):
    for layer in net.layerDict:
        targetLayer = net.layerDict[layer]
        layerOutput = net.getLayerOutput(x, targetLayer)
        print("Layer " + layer + ' l1 loss:', l1loss(layerOutput).cpu().detach().numpy())


def getModule(net, blob):
    modules = blob.split('.')
#    print "Target layer: ", modules
#    if len(modules) == 1:
#        return net._modules.get(blob)
#    else:

    curr_module = net
    print(curr_module)
    for m in modules:
        curr_module = curr_module._modules.get(m)
    return curr_module

def getLayerOutputHook(module, input, output):
    if not hasattr(module, 'activations'):
        module.activations = []
    module.activations.append(output)

def getHookActs(model, module, input):
    if hasattr(module, 'activations'):
        del module.activations[:]
    _ = model.forward(input)
    assert(len(module.activations) == 1)
    return module.activations[0]

def saveImage(img, filepath):
    torchvision.utils.save_image(img, filepath)


def evalTestSplitModel(testloader, netEdge, netCloud, layer, gpu):
    testIter = iter(testloader)
    acc = 0.0
    NBatch = 0
    for i, data in enumerate(testIter, 0):
        batchX, batchY = data
        if gpu:
            batchX = batchX.cuda()
            batchY = batchY.cuda()


        edgeOutput = netEdge.getLayerOutput(batchX, netEdge.layerDict[layer]).clone()
        # except Exception, e:
        #     #print "Except in evalTestSplitModel getLayerOutput, this is a Edge-only model"
        #     #print str(e)
        #     edgeOutput = netEdge.forward(batchX).clone()

        #cloudOuput = net.forward(batchX)
        logits = netCloud.forward_from(edgeOutput, layer)

        #softmax = nn.Softmax().cuda()
        #prob = softmax(logits)
        #print prob[:100,:].max(dim=1)

        if gpu:
            pred = np.argmax(logits.cpu().detach().numpy(), axis = 1)
            groundTruth = batchY.cpu().detach().numpy()
        else:
            pred = np.argmax(logits.detach().numpy(), axis = 1)
            groundTruth = batchY.detach().numpy()
        acc += np.mean(pred == groundTruth)
        NBatch += 1

    accTest = acc / NBatch
    #print "Test accuracy: ", accTest #, "NBatch: ", NBatch, "pred == groundTruth.shape", (pred == groundTruth).shape
    return accTest

def get_PSNR(refimg, invimg, peak = 1.0):
    psnr = 10*np.log10(peak**2 / np.mean((refimg - invimg)**2))
    return psnr
