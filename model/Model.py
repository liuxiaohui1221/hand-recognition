import plotly.offline as pyo
pyo.init_notebook_mode()
import tensorflow as tf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


class Config:
    w, h = 32, 32
    final_class = 6
    MODEL_FILE_NAME = 'handgest_model.h5'
def load_model():
    # load model
    return tf.keras.models.load_model(Config.MODEL_FILE_NAME)

## model
def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )
    return block


def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    return block


def build_model(act, final_class, w, h):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(w, h, 1)),

        tf.keras.layers.Conv2D(16, 3, activation=act, padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation=act, padding='same'),
        tf.keras.layers.MaxPool2D(),

        conv_block(32),
        conv_block(64),

        conv_block(128),
        tf.keras.layers.Dropout(0.2),

        conv_block(256),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),

        tf.keras.layers.Dense(final_class, activation='sigmoid')
    ])
    return model

def saveAndPlot(history , name , model):
    model.save(name)
    epochs = range(1,len(history.history['loss']) + 1)
    epochs = list(epochs)
    Plot(241, epochs, history, 'loss')
    Plot(242, epochs, history, 'accuracy')
    Plot(243,epochs,history,'precision')
    Plot(244,epochs,history,'recall')

    Plot(245, epochs, history, 'val_loss')
    Plot(246, epochs, history, 'val_accuracy')
    Plot(247, epochs, history, 'val_precision')
    Plot(248, epochs, history, 'val_recall')
    plt.show()
def Plot(poc,epochs,history,label):
    plt.subplot(poc)
    y = history.history[label]
    plt.plot(epochs, y, label=label)
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.title(label)
