import torch.nn as nn
from data_provider import DataProvider
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as func
from torchvision import transforms
from tensorboard_logger import configure, log_value
import csv
import scipy.misc


def generate_training_features():
    """
    Loads the saved resnet model and, does a forward pass of the whole Training set, and stores the
    final layer activations as image features into a CSV file
    Arguments:
    None
    """ 
    model = torch.load("models/resnet.model")
    model.eval()
    dp = DataProvider("datasets/training_set")

    batch_size = 500
    batches = dp.no_training_batches(batch_size)

    with open("train_features.csv","w+") as my_csv:
        for i in range(batches):
            print("Iteration: %d/%d" % (i, batches))
            image, label, indices = dp.get_training_batch(batch_size)
            indices = np.reshape(indices, (batch_size, -1))
            images = Variable(torch.from_numpy(image).float()).cuda()
            labels = Variable(torch.from_numpy(label.copy()).long()).cuda()

            # Upsample the images tensor to 240,240
            upsampler = torch.nn.Upsample(size=(224,224), mode='bilinear')
            images = upsampler(images)
            #print(images)
            #print(labels)
            #print("Converted tensors")
            # Forward + Backward + Optimize
            outputs = model(images)
            softmax = torch.nn.Softmax(dim=1)
            outputs = softmax(outputs)
            outputs = outputs.data.cpu().numpy()
            print(np.sum(outputs)) 
            outputs = np.concatenate((indices, outputs), axis=1)
            csvWriter = csv.writer(my_csv,delimiter=',')
            csvWriter.writerows(outputs)


def get_class_predictions():
    """
    Fetches random 50 images from the Test set, and saves the images to disk with the
    the name of their predicted appended to their name
    Arguments:
    None
    """
    model = torch.load("models/resnet.model")
    model.eval()
    dp = DataProvider("datasets/training_set")

    batch_size = 50
    image, label, indices = dp.get_validation_batch(batch_size)
    images = Variable(torch.from_numpy(image).float()).cuda()
    labels = Variable(torch.from_numpy(label.copy()).long()).cuda()

    # Upsample the images tensor to 240,240
    upsampler = torch.nn.Upsample(size=(224,224), mode='bilinear')
    images = upsampler(images)
    #print(images)
    #print(labels)
    #print("Converted tensors")
    # Forward + Backward + Optimize
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)
    predicted = predicted.data.cpu().numpy()
    print(predicted)
    print(predicted.shape)
    for i in range(image.shape[0]):
        name = ""
        if predicted[i] == 0:
            name = "AC Adapters"
        elif predicted[i] == 1:
            name = "Camera Batteries"
        elif predicted[i] == 2:
            name = "Covers"
        elif predicted[i] == 3:
            name = "Keyboards"
        elif predicted[i] == 4:
            name = "Laptops"
        elif predicted[i] == 5:
            name = "Memory"
        elif predicted[i] == 6:
            name = "Point & Shoot Digital Cameras"
        elif predicted[i] == 7:
            name = "USB Cables"
        elif predicted[i] == 8:
            name = "Accessory Kits"
        elif predicted[i] == 9:
            name = "Computers & Accessories"
        elif predicted[i] == 10:
            name = "Headphones"
        elif predicted[i] == 11:
            name = "Lamps"
        elif predicted[i] == 12:
            name = "MP3 Players"
        elif predicted[i] == 13:
            name = "Mice"
        elif predicted[i] == 14:
            name = "Sleeves & Slipcases"
        elif predicted[i] == 15:
            name = "USB Flash Drives"
        else:
            print("lauraaaa")
        name += str(i)
        name = "image_pred_classes/" + name + ".jpg"
        #print("INFO:") 
        #print(image[i].shape)
        im = np.transpose(image[i], (1,2,0))
        print(im.shape)
        scipy.misc.imsave(name, im)

def get_features_vector(image):   
    """
    Returns the extracted feature vector of the passsed image.
    Arguments:
    - An image array
    """ 
    model = torch.load("models/resnet.model")
    model.eval()
    image = np.transpose(image, (2,0,1))
    im = np.reshape(image, (-1, image.shape[0], image.shape[1], image.shape[2]))
    images = Variable(torch.from_numpy(im).float()).cuda()
    upsampler = torch.nn.Upsample(size=(224,224), mode='bilinear')
    images = upsampler(images)
    out = model(images)
    softmax = torch.nn.Softmax(dim=1)
    out = softmax(out)
    return out.data.cpu().numpy()

#get_class_predictions()
#generate_training_features()
  
