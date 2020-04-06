
import re
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(shuffle=True):
    data = {i: [] for i in range(10)}

    for jpg in glob.glob('data/digits/*.jpg'):
        m = re.search(r'd([\d_])\.jpg$', jpg)
        if m[1] == '_':
            label = 0
        else:
            label = int(m[1])
        img = cv2.imread(jpg, 0)
        img = cv2.bitwise_not(img)
        data[label].append(img)

    # 空白マスの画像を削減する
    if len(data[0]) > 30:
        data[0] = random.sample(data[0], 30)

    X_train = [None] * 10
    X_test  = [None] * 10
    y_train = [None] * 10
    y_test  = [None] * 10

    for i in range(0, 10):
        # print(i, len(data[i]))
        X_train[i], X_test[i], y_train[i], y_test[i] = \
            train_test_split(data[i], [i]*len(data[i]), test_size=0.2)

    X_train = np.array(list(itertools.chain.from_iterable(X_train))) # flatten
    X_test  = np.array(list(itertools.chain.from_iterable(X_test)))  # flatten
    y_train = np.array(list(itertools.chain.from_iterable(y_train))) # flatten
    y_test  = np.array(list(itertools.chain.from_iterable(y_test)))  # flatten

    if shuffle:
        p = np.random.permutation(len(X_train))
        X_train = X_train[p]
        y_train = y_train[p]
        p = np.random.permutation(len(X_test))
        X_test = X_test[p]
        y_test = y_test[p]

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.1)

X_train = X_train.reshape(X_train.shape + (1,))
X_valid = X_valid.reshape(X_valid.shape + (1,))
X_test  = X_test .reshape(X_test .shape + (1,))

batch_size = 20
total_train = X_train.shape[0]
total_val   = X_valid.shape[0]

print('total_train:', total_train)
print('total_val:', total_val)


image_gen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=.05,
    height_shift_range=.05)

train_data_gen = image_gen.flow(
    X_train, y_train,
    batch_size=batch_size,
    shuffle=True)

image_gen_val = ImageDataGenerator()

val_data_gen = image_gen_val.flow(
    X_valid, y_valid,
    batch_size=batch_size)

# def show_img(x, y=None, rows=1):
#     n = len(x)
#     if y is None:
#         y = [None] * n
#     plt.figure(figsize=(10,2*rows))
#     for i, (img, label) in enumerate(zip(x, y)):
#         plt.subplot(rows, n//rows, i+1)
#         plt.axis('off')
#         if label is not None:
#             plt.title(label2text[int(label)])
#         img = img.reshape(64, 64)
#         print(img.shape)
#         print(img)
#         plt.imshow(img, 'gray')
#     plt.show()
#
# augmented_images = [train_data_gen[0][0][0] for i in range(10)]
# show_img(augmented_images, rows=2)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten()) # 次元数を3から1にする
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) # 10種類に分類する

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=50,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

# model.save('my_digit_model.h5')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# # 予想
# img = cv2.imread('data/digits/f08-n53-d4.jpg', 0)
# img = cv2.bitwise_not(img)
# target = np.array([img])
# target = target.reshape(target.shape + (1,))
#
# result = model.predict(target)
# predicted_class = np.argmax(result[0], axis=-1)
# print(predicted_class)
