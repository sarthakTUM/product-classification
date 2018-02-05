import tables
import numpy as np
import os


class BatchGenerator:
    def __init__(self, train_images, train_labels):
        self._timages = train_images
        self._tlabels = train_labels

        self.train_offset = 0
        self.val_offset = 0
        
        self.tr_size = int(np.floor(self._timages.shape[0]*0.7)) # 70% of the training set as training 
        
        self.indices_train = np.arange(self._timages.shape[0])

        np.random.shuffle(self.indices_train)

        self.indices_val = self.indices_train[self.tr_size:self._timages.shape[0]]
        self.indices_train = self.indices_train[0:self.tr_size]

    def get_next_training_batch(self, batch_size):
        images = []
        labels = []
        if (self.train_offset + batch_size) <= len(self.indices_train):
            images = self._timages[self.indices_train[self.train_offset:self.train_offset + batch_size], :]
            labels = self._tlabels[self.indices_train[self.train_offset:self.train_offset + batch_size]]
            indices = self.indices_train[self.train_offset:self.train_offset + batch_size]
            self.train_offset += batch_size
        else:
            images = self._timages[self.indices_train[self.train_offset:],:]
            labels = self._tlabels[self.indices_train[self.train_offset:]]
            indices = self.indices_train[self.train_offset:]
            self.train_offset += batch_size
            
        if self.train_offset >= len(self.indices_train):
            print("Training epoch happening after this return")
            # An epoch happened on train data
            # shuffle Training data indices
            np.random.shuffle(self.indices_train)
            self.train_offset = 0
        return images, labels, indices

    def get_next_validation_batch(self, batch_size):
        images = []
        labels = []
        if (self.val_offset + batch_size) <= len(self.indices_val):
            images = self._timages[self.indices_val[self.val_offset:self.val_offset + batch_size], :]
            labels = self._tlabels[self.indices_val[self.val_offset:self.val_offset + batch_size]]
            indices = self.indices_val[self.val_offset:self.val_offset + batch_size]
            self.val_offset += batch_size
        else:
            images = self._timages[self.indices_val[self.val_offset:],:]
            labels = self._tlabels[self.indices_val[self.val_offset:]]
            indices = self.indices_val[self.val_offset:]
            self.val_offset += batch_size
            
        if self.val_offset >= len(self.indices_val):
            print("Validation epoch happening after this return")
            # An epoch happened on val data
            # shuffle VALIDATION data indices
            np.random.shuffle(self.indices_val)
            self.val_offset = 0
        return images, labels, indices
    


# In[4]:


class DataProvider:
    """
    Class which provides data related functions
    """

    def __init__(self, trainhdf5file):
        trainhdf5file = os.path.abspath(trainhdf5file)

        trainhdf5file = tables.open_file(trainhdf5file, "r")
        print(trainhdf5file)
        
        self.train_X = trainhdf5file.root.images
        self.train_Y = trainhdf5file.root.labels
        print("shape is %d")
        print( self.train_X.shape)
        labels = np.array(self.train_Y)
        labels = labels - 1
        #for label in self.train_Y:
        #    lb = np.zeros(15) # Wir haben nur funfzehn categorin
        #    lb[label-1] = 1
        #    labels.append(lb)
        #labels = np.array(labels)
        print(labels.shape)

        # To make everything fast at the expense of huge RAM usage, pass these handlers as numpy arrays 
        # to BatchGenerator
        print("Initializing bath generator")
        self.batch_handler = BatchGenerator(np.array(self.train_X), labels)
        print("Initialized batch generator")
    def get_training_batch(self, n):
        return self.batch_handler.get_next_training_batch(n)

    def no_training_batches(self, batch_size):
        return int(np.ceil(float(len(self.batch_handler.indices_train)/float(batch_size))))
    
    def no_validation_batches(self, batch_size):
        return int(np.ceil(len(self.batch_handler.indices_val)/float(batch_size)))
        
    def get_validation_batch(self, n):
        return self.batch_handler.get_next_validation_batch(n)

