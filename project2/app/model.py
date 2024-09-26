
import os
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

from project2.app.metrics import dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing

class Unet:
    def __init__(self, img_size, num_classes, ker_init='he_normal', dropout=0.2, learning_rate=0.001):
        self.img_size = img_size
        self.num_classes = num_classes
        self.ker_init = ker_init
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        inputs = Input((self.img_size, self.img_size, 2))
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(inputs)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv1)

        pool = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(pool)
        conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv3)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(pool4)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv5)
        drop5 = Dropout(self.dropout)(conv5)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=self.ker_init)(UpSampling2D(size=(2, 2))(drop5))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=self.ker_init)(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=self.ker_init)(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv9)

        up = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=self.ker_init)(UpSampling2D(size=(2, 2))(conv9))
        merge = concatenate([conv1, up], axis=3)
        conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(merge)
        conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=self.ker_init)(conv)

        conv10 = Conv2D(self.num_classes, (1, 1), activation='softmax')(conv)

        return Model(inputs=inputs, outputs=conv10)

    def compile_model(self, loss="categorical_crossentropy"):
        metrics = [
            'accuracy', 
            tf.keras.metrics.MeanIoU(num_classes=4), 
            dice_coef, 
            precision, 
            sensitivity, 
            specificity, 
            dice_coef_necrotic, 
            dice_coef_edema, 
            dice_coef_enhancing
        ]
        self.model.compile(loss=loss,
                           optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                           metrics=metrics)

    def plot_model(self, file_path='unet_model.png'):
        plot_model(self.model, show_shapes=True, show_layer_names=True, to_file=file_path)

    def train(self, training_generator, validation_generator, epochs=35, train_ids=None):
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1),
            ModelCheckpoint(filepath='/home/jupyter/model_.{epoch:02d}-{val_loss:.6f}.weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
            CSVLogger('training.log', separator=',', append=False)
        ]

        K.clear_session()
        history = self.model.fit(
            training_generator,
            epochs=epochs,
            steps_per_epoch=len(train_ids),
            callbacks=callbacks,
            validation_data=validation_generator
        )
        return history

    def save_model(self, file_path='my_model.keras'):
        self.model.save(file_path)

    def load_model(self, file_path='my_model.keras', custom_objects=None):
        if custom_objects is None:
            custom_objects = {
                "accuracy": tf.keras.metrics.MeanIoU(num_classes=self.num_classes)
            }
        self.model = keras.models.load_model(file_path, custom_objects=custom_objects, compile=False)

