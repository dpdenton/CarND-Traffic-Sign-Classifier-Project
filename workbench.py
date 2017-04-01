# Load pickled data
import cv2
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import groupby

# TODO: Fill this in based on where you saved the training and testing data

training_file = "data/train.p"
validation_file= "data/valid.p"
testing_file = "data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# greyscale
X_train = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X_train])
X_train = X_train.reshape( (len(X_train), 32, 32, 1) )

fname = 'signnames.csv'
with open(fname) as f:
    content = f.readlines()
    #
    # for l in f.readlines():
    #     if l.startswith('17'):
    #         print("Sign deteceted: {}".format(l))

sign = next(l for l in content if l.startswith('17'))
print(sign)

# you may also want to remove whitespace characters like `\n` at the end of each line
# content = [x.strip() for x in content]
# print(content)