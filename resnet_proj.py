from data_provider import DataProvider
import numpy as np
#import matplotlib.pyplot as plt
from classif_model import ClassifNN
from solver import Solver

# Get the data provider
dp = DataProvider("datasets/training_set")
# Define the model
model = ClassifNN()
# Train the model
batch_size = 100
solver = Solver(dp)
solver.train(model, batch_size)

model.save("models/resnet.model")

