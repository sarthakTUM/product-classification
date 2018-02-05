from random import shuffle
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as func
from torchvision import transforms
from tensorboard_logger import configure, log_value

class Solver(object):
    default_adam_args = {"lr": 9e-3,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, data_provider, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.data_provider = data_provider
        self.loss_func = loss_func
        self._reset_histories()
        configure("logger/logs") # Configure tensorboard logger

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def do_train_stuff(self, batches, batch_size, log_nth, optim, model, epoch):
        # Calculate the training loss and accuracy
        outputs = 0
        with torch.cuda.device(0):
            for i in range((epoch*batches), ((epoch+1)*batches)):#for i in range(batches):
                image, label = self.data_provider.get_training_batch(batch_size)
                
                images = Variable(torch.from_numpy(image).float()).cuda()
                labels = Variable(torch.from_numpy(label.copy()).long()).cuda()

                # Upsample the images tensor to 240,240
                upsampler = torch.nn.Upsample(size=(224,224), mode='bilinear')
                images = upsampler(images)
                #print(images)
                #print(labels)
                #print("Converted tensors")
                optim.zero_grad()
                # Forward + Backward + Optimize
                outputs = model(images)

                loss =  self.loss_func(outputs, labels)
                log_value("training_minibatch_loss", loss.data[0], i)
                loss.backward()
                optim.step()

                # Calculate accuracy of the batch
                # Storing values
                self.train_loss_history.append(loss.data[0])

                # calculate accuracy
                _, train_predicted = torch.max(outputs, 1) 
                train_accuracy = np.mean((train_predicted == labels).data.cpu().numpy())
                # Loss is logged every nth iteration
                if (i+1) % log_nth == 0:
                    print ('[Iteration %d/%d] TRAIN loss: %0.4f, acc: %0.4f' % (i, batches, loss.data[0], train_accuracy))

                self.train_acc_history.append(train_accuracy)
                log_value("Training_minibatch_acc", train_accuracy, i)
                val_accuracy, val_loss = self.do_val_stuff2(batch_size, model)
                log_value("Validation_minibatch_loss", val_loss, i)
                log_value("Validation_minibatch_acc", val_accuracy, i)

            # Epoch happened, print stats
            print("saving model after epoch %d" % epoch)
            model.save("models/resnet.model")
            print ('[Epoch %d] Train acc/loss: %0.4f/%0.4f Val acc/loss: %0.4f/%0.4f' % (epoch, self.train_acc_history[-1], loss.data[0],
                                                                            val_accuracy, val_loss))

    def do_val_stuff2(self, batch_size, model):
        image, label = self.data_provider.get_validation_batch(batch_size)
        images = Variable(torch.from_numpy(image).float()).cuda()
        labels = Variable(torch.from_numpy(label.copy()).long()).cuda()

        # Upsample the images tensor to 240,240
        upsampler = torch.nn.Upsample(size=(224,224), mode='bilinear')
        images = upsampler(images)

        # Forward + Backward + Optimize
        outputs = model(images)

        loss = self.loss_func(outputs, labels)
        # Calculate accuracy of the batch
        # Storing values
        self.val_loss_history.append(loss.data[0])

        _, val_predicted = torch.max(outputs, 1)
        labels_mask = labels >= 0
        val_accuracy =  np.mean((val_predicted == labels).data.cpu().numpy())

        self.val_acc_history.append(val_accuracy)

        print ('Val acc/loss: %0.4f/%0.4f' % (val_accuracy, loss.data[0]))

        return val_accuracy, loss.data[0]

    def train(self, model, batch_size, num_epochs=10, log_nth=1):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """

        # Get layers for which requires_grad is true, i.e which need to be optimized
        gradable_params = filter(lambda x: x.requires_grad, model.parameters())
        optim = self.optim(gradable_params, **self.optim_args)

        self._reset_histories()

        if torch.cuda.is_available():
           
            print(torch.cuda.current_device())
            print(torch.cuda.device_count())
            with torch.cuda.device(0):
                print("changed device to:")
                print(torch.cuda.current_device())
                model.cuda(0)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        no_train_batches = self.data_provider.no_training_batches(batch_size)
        #no_val_batches = self.data_provider.no_validation_batches(batch_size)
        for epoch in range(num_epochs):
            self.do_train_stuff(no_train_batches, batch_size, log_nth, optim, model, epoch)
            #self.do_val_stuff(no_val_batches, batch_size, optim, model, epoch)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')

