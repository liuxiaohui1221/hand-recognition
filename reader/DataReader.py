import pandas as pd
import os
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
def Create_Directory_DataFrame(basedir):
    df =pd.DataFrame(columns=['Class','Location'])
    for folder in os.listdir(basedir):
        for Class in os.listdir(basedir+folder+'/'):
            for location in os.listdir(basedir+folder+'/'+Class+'/'):
                df = df.append({'Class':Class,'Location':basedir+folder+'/'+Class+'/'+location},ignore_index=True)
    df = df.sample(frac = 1)
    return df
def read_train_test_data(df,w,h):
    X,y,enc=handle_data(df,w,h)
    # segmentation train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
    return X_train, X_test, y_train, y_test, enc

def handle_data(df,w,h):
    train_image = []
    for location in tqdm(df.iloc[:]['Location']):
        img = cv2.imread(location, 0)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = img.reshape(w, h, 1)
        train_image.append(img)
    X = np.array(train_image)

    y = np.array(df.iloc[:]['Class'])
    y = y.reshape(y.shape[0], 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y)
    print(enc.categories_)
    y = enc.transform(y).toarray()
    print('Data   :   ' + str(X.shape))
    print('Output :   ' + str(y.shape))
    return X,y,enc

def read_Hand_Gest_DataFrame(traindir,testdir,w,h):
    df_train = pd.DataFrame(columns=['Class', 'Location'])
    for Class in os.listdir(traindir):
        for location in os.listdir(traindir + Class + '/'):
            df_train=df_train.append({'Class':Class,'Location':traindir+Class+'/'+location},ignore_index=True)
    df_train = df_train.sample(frac=1)

    df_test = pd.DataFrame(columns=['Class', 'Location'])
    for Class in os.listdir(testdir):
        for location in os.listdir(testdir + Class + '/'):
            df_test = df_test.append({'Class': Class, 'Location': testdir + Class + '/' + location}, ignore_index=True)
    df_test = df_test.sample(frac=1)

    X_train, y_train,enc1 = handle_data(df_train, w, h)
    X_test, y_test,enc_test = handle_data(df_test, w, h)
    return X_train, X_test, y_train, y_test, enc_test

def readImageToDataFrame(Class,imgPath):
    df_test = pd.DataFrame(columns=['Class', 'Location'])
    df_test = df_test.append({'Class': Class, 'Location': imgPath}, ignore_index=True)
    X_test, y_test, enc_test = handle_data(df_test, 64, 64)
    return X_test, y_test, enc_test
def handle_img(img,w,h):
    train_image = []
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = img.reshape(w, h, 1)
    train_image.append(img)
    X = np.array(train_image)
    return X