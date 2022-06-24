import plotly.offline as pyo
pyo.init_notebook_mode()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DataReader
import cv2
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn import preprocessing
import random
import tensorflow as tf
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import Model
warnings.filterwarnings("ignore")
tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


df = DataReader.Create_Directory_DataFrame('./input/leapgestrecog/leapgestrecog/')
print(df.shape)
df.head()

count = 1
f = plt.figure(figsize=(50,13))
for Class in df['Class'].unique():
    seg = df[df['Class']==Class]
    address =  seg.sample().iloc[0]['Location']
    img = cv2.imread(address,0)
    ax = f.add_subplot(2, 5,count)
    ax = plt.imshow(img)
    ax = plt.title(Class,fontsize= 30)
    count = count + 1
plt.suptitle("Hand Sign Images", size = 32)
#plt.show()

w , h= 64,64
final_class = 10
# data creation
from tqdm import tqdm
train_image = []
for location in tqdm(df.iloc[:]['Location']):
    img = cv2.imread(location,0)
    img = cv2.resize(img, (w,h), interpolation = cv2.INTER_AREA)
    img = img.reshape(w,h,1)
    train_image.append(img)
X = np.array(train_image)

y = np.array(df.iloc[:]['Class'])
y = y.reshape(y.shape[0],1)
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y)
print(enc.categories_)
y = enc.transform(y).toarray()
print('Data   :   '+str(X.shape))
print('Output :   '+str(y.shape))

plt.figure(figsize=(25,8))
plt.imshow(X[66].reshape(w,h))
plt.title(enc.inverse_transform(y[0].reshape(1,10))[0][0],size = 20)
#plt.show()

#segmentation train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)


def wrap(Training_Output_Results , Opt , Act ,  history):
    epoch  = len(history.history['loss'])
    epochs = list(np.arange(1,epoch + 1,1))
    Optimizer = np.repeat(Opt,epoch).tolist()
    Activation = np.repeat(Act,epoch).tolist()
    cumiliated_res = {}
    cumiliated_res['Epochs']=epochs
    cumiliated_res['Optimizer']=Optimizer
    cumiliated_res['Activation_Function']=Activation
    cumiliated_res['Train_Loss']=history.history['loss']
    cumiliated_res['Train_Accuracy']=history.history['accuracy']
    cumiliated_res['Train_Precision']=history.history['precision']
    cumiliated_res['Train_Recall']=history.history['recall']
    cumiliated_res['Val_Loss']=history.history['val_loss']
    cumiliated_res['Val_Accuracy']=history.history['val_accuracy']
    cumiliated_res['Val_Precision']=history.history['val_precision']
    cumiliated_res['Val_Recall']=history.history['val_recall']
    convertDictionary = pd.DataFrame(cumiliated_res)
    Training_Output_Results = Training_Output_Results.append(convertDictionary)
    return Training_Output_Results

Optimisers = ['RMSprop','Adam','Adadelta','Adagrad']
Activation_function =['relu','sigmoid','softmax','tanh','softsign','selu','elu']

Training_Output_Results = pd.DataFrame(
    columns=['Epochs', 'Optimizer', 'Activation_Function', 'Train_Loss', 'Train_Accuracy', 'Train_Precision',
             'Train_Recall', 'Val_Loss', 'Val_Accuracy', 'Val_Precision', 'Val_Recall'])


def Optimise_verify(Training_Output_Results):
    for opt in Optimisers:
        model = Model.build_model(Activation_function[0], final_class, w, h)
        METRICS = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=METRICS
        )
        history = model.fit(X_train, y_train, epochs=10, validation_split=0.3, batch_size=15, verbose=0, shuffle=True)
        Training_Output_Results = wrap(Training_Output_Results, opt, Activation_function[0], history)
        print('---------------------Round for ' + opt + ' Completed-----------------------------------------')
    return Training_Output_Results
Training_Output_Results = Optimise_verify(Training_Output_Results)
Training_Output_Results=Training_Output_Results.sample(frac = 1)
print(Training_Output_Results.shape)
Training_Output_Results.to_csv('Optimizer_64_64_data.csv', index = False)
Training_Output_Results.head()

Training_Output_Results = pd.DataFrame(
    columns=['Epochs', 'Optimizer', 'Activation_Function', 'Train_Loss', 'Train_Accuracy', 'Train_Precision',
             'Train_Recall',
             'Val_Loss', 'Val_Accuracy', 'Val_Precision', 'Val_Recall'])

def Activation_verify(Training_Output_Results):
    for act in Activation_function:
        model = Model.build_model(act, final_class, w, h)
        METRICS = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
        model.compile(
            optimizer=Optimisers[0],
            loss='categorical_crossentropy',
            metrics=METRICS
        )
        history = model.fit(X_train, y_train, epochs=10, validation_split=0.3, batch_size=15, verbose=0, shuffle=True)
        Training_Output_Results = wrap(Training_Output_Results, Optimisers[0], act, history)
        print('---------------------Round for ' + act + ' Completed-----------------------------------------')
    return Training_Output_Results
Training_Output_Results = Activation_verify(Training_Output_Results)
Training_Output_Results=Training_Output_Results.sample(frac = 1)
print(Training_Output_Results.shape)

Training_Output_Results.to_csv('Activation_64_64_data.csv', index = False)
Training_Output_Results.head()