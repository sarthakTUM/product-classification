import torch
import torch.nn as nn
import torchvision

# Class that encapsualtes a ResNet model
class ClassifNN(nn.Module):

    def __init__(self):
        super(ClassifNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        #args = {"num_classes": 15}
        self.model = torchvision.models.resnet152(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        #print("Number of layers is %d" % len(self.model.parameters()))
        in_features = self.model.fc.in_features
        # Change the last layer
        self.model.fc = nn.Linear(in_features, 16)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        out = self.model(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return out

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
