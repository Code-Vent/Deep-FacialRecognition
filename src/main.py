import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, dataset
from  torchvision import transforms, datasets
from torchvision.models import resnet50
import numpy as np
import cv2 as cv
import numpy as np
import random
import os
from matplotlib import pyplot as plt
import uuid
#import mmdet.engine as eng
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall


import data_collection
from data_collection import ANC_PATH, POS_PATH, NEG_PATH



anchor = tf.data.Dataset.list_files('{}*.jpg'.format(ANC_PATH)).take(300)
positive = tf.data.Dataset.list_files('{}*.jpg'.format(POS_PATH)).take(300)
negative = tf.data.Dataset.list_files('{}*.jpg'.format(NEG_PATH)).take(300)

positive_loader = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negative_loader = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data_loader = positive_loader.concatenate(negative_loader)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(250),
    transforms.ToTensor(),
])

#anchor = datasets.ImageFolder(ANC_PATH, transform, allow_empty=True)
#anchor_loader = DataLoader(anchor, batch_size=4, shuffle=True)


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100,100))
    img = img / 255.0
    return img

def preprocess_twin(input_img, val_img, label):
    return (preprocess(input_img), preprocess(val_img), label)

data_loader = data_loader.map(preprocess_twin)
data_loader = data_loader.cache()
data_loader = data_loader.shuffle(buffer_size=1024)

train_data = data_loader.take(round(len(data_loader)*0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

test_data = data_loader.skip(round(len(data_loader)*0.7))
test_data = data_loader.take(round(len(data_loader)*0.3))
test_data = train_data.batch(16)
test_data = train_data.prefetch(8)

train_samples = train_data.as_numpy_iterator()
test_samples = test_data.as_numpy_iterator()


def make_embedding():
    inp = Input(shape=(100,100,3), name='input_image')
    c1 = Conv2D(64,(10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64,(2,2),padding='same')(c1)
    c2 = Conv2D(128,(7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64,(2,2),padding='same')(c2)
    
    c3 = Conv2D(128,(4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64,(2,2),padding='same')(c3)
    
    c4 = Conv2D(256,(4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return Model(inputs=[inp], outputs=d1, name='embedding')


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, input_embedding, validation_embedding):
        print(input_embedding)
        return tf.math.abs(input_embedding - validation_embedding)

embedding = make_embedding()
embedding.summary()    

    
def make_siamese_model():
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='siameseNetwork')

siamese_model = make_siamese_model()
siamese_model.summary()

binary_cross_loss = tf.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=optimizer, siamese_model=siamese_model)

@tf.function
def train_step(batch):
    
    with tf.GradientTape() as tape:
        x = batch[:2]
        y_true = batch[2]
        y_pred = siamese_model(x, training=True)
        loss = binary_cross_loss(y_true, y_pred)
    print(loss)    
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    optimizer.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss


def train(data, EPOCHS):
    for epoch in range(EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        prog_bar = tf.keras.utils.Progbar(len(data))
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            prog_bar.update(idx)
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            
            
EPOCHS = 15
train(train_data, EPOCHS)

test_input, test_val, y_true = test_samples.next()
y_pred = siamese_model.predict([test_input, test_val])
y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]

m = Recall()
m.update_state(y_true, y_pred)
m.result().numpy()

siamese_model.save('siamesemodel.h5')

model = tf.keras.models.load_model('siamesemodel.h5',
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})

def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('../application_data', 'verification_images')):
        input_img = preprocess(os.path.join('../application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('../application_data', 'verification_images', image))
        
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
        
    detection = np.sum(np.array(results) > detection_threshold)
    
    verification = detection / len(os.listdir(os.path.join('../application_data', 'verification_images')))
    verified = verification > verification_threshold
    
    return results, verified


from data_collection import PATHS, capture_anchor_and_positive

PATHS[ord('v')] = '../application_data/input_image/'

def menu():
    print('Press \'v\' to start Verification')
    print('Press \'q\' to stop Verification')

while True:
    menu()
    capture_anchor_and_positive(filename='input_image', ext_key=ord('v'))
    results, verified = verify(model, 0.5, 0.5)
    print(verified)
    ans = input('Try again? (Y/N) ')
    if ans.lower() == 'n':
        break


