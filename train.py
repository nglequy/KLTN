import numpy as np
import tensorflow as tf
import os
import cv2
from keras import Model
from keras.layers import Dense, Flatten, Dropout
from keras.applications.vgg16 import VGG16
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

models_path = "/content/drive/MyDrive/SignLanguage/models"
train_path = "/content/drive/MyDrive/SignLanguage/ASL_data/ASL_data/training_set"
test_path = "/content/drive/MyDrive/SignLanguage/ASL_data/ASL_data/test_set"

image_path = '/content/drive/MyDrive/SignLanguage/data_Feb_28/data'
filepath = "weights{epoch:02d}{val_accuracy:.2f}.hdf5"

w, h = 224, 224


class Dataset:
    def __init__(self, data, label, w, h):
        # the paths of images
        self.data = np.array(data)
        # the paths of segmentation image

        # binary encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(label)

        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        self.label = onehot_encoded
        self.w = w
        self.h = h

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.data[i])
        image = cv2.resize(image, (self.w, self.h))
        label = self.label[i]
        return image, label


class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.size = size

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return tuple(batch)

    def __len__(self):
        return self.size // self.batch_size


def load_path(path):
    data = []
    label = []

    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            file_path = os.path.join(path, folder, file)
            data.append(file_path)
            label.append(folder)
    return data, label


# Image_train, Label_train = load_path(train_path)
# Image_test, Label_test = load_path(test_path)

Image, Label = load_path(image_path)
Image_train, Image_test, Label_train, Label_test = train_test_split(Image, Label, test_size=0.2, random_state=12)

# Build dataaset
train_dataset = Dataset(Image_train, Label_train, w, h)
test_dataset = Dataset(Image_test, Label_test, w, h)

# Loader
train_loader = Dataloader(train_dataset, 8, len(train_dataset))
test_loader = Dataloader(test_dataset, 8, len(test_dataset))

model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [model_checkpoint]

models = VGG16(weights='imagenet',
               include_top=False,
               input_shape=(224, 224, 3))
optimizers = tf.keras.optimizers.Adam()

x = models.output
x = Flatten()(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc2a')(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', name='fc4')(x)

predictions = Dense(30, activation='softmax')(x)
model = Model(inputs=models.input, outputs=predictions)

for layer in models.layers:
    layer.trainable = False

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_loader,
                    validation_data=test_loader,
                    epochs=50,
                    verbose=1,
                    callbacks=[callbacks_list])

model.save('/content/drive/MyDrive/SignLanguage/model/train_model_Feb28_1.h5')
