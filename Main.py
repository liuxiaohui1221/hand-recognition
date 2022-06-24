import DataReader
import Model
import tensorflow as tf
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
w , h= 64,64
final_class = 10
model = Model.build_model('relu', final_class ,w , h)

# 打印模型概述信息
model.summary()

METRICS = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
]
model.compile(
                optimizer='RMSprop',
                loss='categorical_crossentropy',
                metrics=METRICS
        )

# data creation
df = DataReader.Create_Directory_DataFrame('./input/leapgestrecog/leapgestrecog/')
print(df.shape)
df.head()
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
#segmentation train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

# train
# history = model.fit(X_train, y_train, epochs=50, validation_split=0.3, batch_size=15,verbose=1,shuffle=True)
# Model.saveAndPlot(history , 'final_model',model)

# load model
model = tf.keras.models.load_model('final_model.h5')

# test
y_pred = model.evaluate(X_test , y_test, verbose = True)
print(y_pred)

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes=True)

y_prediction = model.predict(X_test)
def binary_classify(y_pred):
    for inp in y_pred:
        maximum = 0
        index = 0
        for i in range(10):
            if(maximum != max(maximum,inp[i])):
                maximum = max(maximum,inp[i])
                index = i
            inp[i] = 0
        inp[index]=1
    return y_pred
y_prediction  = binary_classify(y_prediction)


def create_result(y):
    y_final = []
    for i in range(y.shape[0]):
        y_final.append(enc.inverse_transform(y[i].reshape(1, 10))[0][0])
    return y_final


def remove_none(y, y_pred):
    index = []
    for i in range(len(y) - 1, 0, -1):
        if y_pred[i] == None:
            del y[i]
            del y_pred[i]

    return y, y_pred


def label_encode(y, y_pred):
    le = preprocessing.LabelEncoder()
    le.fit(y_pred)
    print(le.classes_)
    y = le.transform(y)
    y_pred = le.transform(y_pred)
    return y, y_pred


y_class_result = create_result(y_prediction)
y_class_desired = create_result(y_test)
y_label_desired , y_label_result = label_encode(y_class_desired , y_class_result)

from sklearn.metrics import classification_report
tn = []
for cat in enc.categories_[0].reshape(10,1):
    tn.append(cat[0])
target_names = tn
print(classification_report(y_label_desired, y_label_result, target_names=target_names))