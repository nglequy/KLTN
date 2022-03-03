import numpy as np
import tensorflow as tf
import os
import cv2
from keras import Model
from keras.layers import Dense, Flatten, Dropout
from keras.applications.vgg16 import VGG16
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

models_path = "/content/drive/MyDrive/SignLanguage/models"
train_path = "/content/drive/MyDrive/SignLanguage/ASL_data/training_set"
test_path = "/content/drive/MyDrive/SignLanguage/ASL_data/test_set"
filepath = "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"

w, h = 224, 224


# Dung de tao toan bo du lieu va load theo batch
class Dataset:
    def __init__(self, data, label, w, h):
        # the paths of images
        self.data = np.array(data)
        # the paths of segmentation images

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


Image_train, Label_train = load_path(train_path)
Image_test, Label_test = load_path(test_path)

# Build dataaset
train_dataset = Dataset(Image_train, Label_train, w, h)
test_dataset = Dataset(Image_test, Label_test, w, h)

# Loader

train_loader = Dataloader(train_dataset, 8, len(train_dataset))
test_loader = Dataloader(test_dataset, 8, len(test_dataset))

model_checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
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

predictions = Dense(26, activation='softmax')(x)
model = Model(inputs=models.input, outputs=predictions)

for layer in models.layers:
    layer.trainable = False

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

result = model.fit_generator(train_loader,
                             validation_data=test_loader,
                             epochs=50,
                             verbose=1,
                             callbacks=[callbacks_list])

result.save('/content/drive/MyDrive/SignLanguage/model/train_model.h5')

# import tensorflow as tf
# import numpy as np
# import os
# from PIL import Image
# from PIL import ImageFile
# from PIL.ImageQt import rgb
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.vgg16 import VGG16
# from keras import Model
# from keras.layers import Dense, Dropout, Flatten
# from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.python.keras.utils.np_utils import to_categorical
#
# ImageFile.LOAD_TRUNCATED_IMAGES = True
#
# models_path = "./models"
# train_path = "./data/training_set"
# test_path = "./data/test_set"
# train_labels = os.listdir(train_path)
# test_labels = os.listdir(test_path)
#
#
#
# def resize_image(file):
#     img = Image.open(file)
#     img = img.resize((224, 224))
#     img = np.array(img)
#     return img
#
#
# def process_data(image_data, label_data):
#     image_data = np.array(image_data, dtype='float32')
#     if rgb:
#         pass
#     else:
#         image_data = np.stack((image_data,) * 3, axis=-1)
#     image_data /= 255
#     label_data = np.array(label_data)
#     label_data = to_categorical(label_data)
#     return image_data, label_data
#
#
# def load_data(path, labels):
#     image_data = []
#     label_data = []
#
#     for name in labels:
#         dir = os.path.join(path, name)
#         current_label = name
#         for x in range(1, 1300):
#             file = dir + "/" + str(x) + ".png"
#             # current_image = cv2.imread(file)
#             image_data.append(resize_image(file))
#             label_data.append(current_label)
#     image_data, label_data = process_data(image_data, label_data)
#     return image_data, label_data
#
#
# Image_train, Label_train = load_data(train_path, train_labels)
# Image_test, Label_test = load_data(test_path, test_labels)
#
# model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True)
# early_stopping = EarlyStopping(monitor='val_acc',
#                                min_delta=0,
#                                patience=10,
#                                verbose=1,
#                                mode='auto',
#                                restore_best_weights=True)
#
# models = VGG16(weights='imagenet',
#                include_top=False,
#                input_shape=(224, 224, 3))
# optimizers = tf.keras.optimizers.Adam()
#
# x = models.output
# x = Flatten()(x)
# x = Dense(128, activation='relu', name='fc1')(x)
# x = Dense(128, activation='relu', name='fc2')(x)
# x = Dense(128, activation='relu', name='fc2a')(x)
# x = Dense(128, activation='relu', name='fc3')(x)
# x = Dropout(0.5)(x)
# x = Dense(64, activation='relu', name='fc4')(x)
#
# predictions = Dense(5, activation='softmax')(x)
# model = Model(input=models.input, outputs=predictions)
#
# for layer in models.layers:
#     layer.trainable = False
#
# model.compile(optimizer='Adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(Image_train, Label_train,
#           epochs=50, batch_size=64,
#           validation_data=(Image_test, Label_test),
#           verbose=1,
#           callbacks=[early_stopping, model_checkpoint])
#
# model.save('model/train_model.h5')


from glob import glob

from PIL import ImageFile
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator

# ImageFile.LOAD_TRUNCATED_IMAGES = True
#
# models_path = "/content/drive/MyDrive/SignLanguage/models"
# # train_path = "/content/drive/MyDrive/SignLanguage/training"
# # test_path = "/content/drive/MyDrive/SignLanguage/test"
# train_path = "./data/training_set"
# test_path = "./data/test_set"
#
# IMAGE_SIZE = [224, 224]
#
# inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# for layer in inception.layers:
#     layer.trainable = False
# folders = glob('./data/training_set/*')
# x = Flatten()(inception.output)
# prediction = Dense(len(folders), activation='softmax')(x)
# model = Model(inputs=inception.input, outputs=prediction)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# train_datagen = ImageDataGenerator(rescale=1. / 255,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True)
#
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# training_set = train_datagen.flow_from_directory('./data/training_set',
#                                                  target_size=(224, 224),
#                                                  batch_size=16,
#                                                  class_mode='categorical')
#
# test_set = test_datagen.flow_from_directory('./data/test_set',
#                                             target_size=(224, 224),
#                                             batch_size=16,
#                                             class_mode='categorical')
#
# r = model.fit_generator(training_set,
#                         validation_data=test_set,
#                         epochs=50,
#                         steps_per_epoch=len(training_set),
#                         validation_steps=len(test_set))
