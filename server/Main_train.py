from reader import DataReader
from model import Model
import tensorflow as tf
from sklearn import preprocessing
from model.Model import Config

tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = Model.build_model('relu', Config.final_class, Config.w, Config.h)

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
# df = DataReader.Create_Directory_DataFrame('./input/leapgestrecog/leapgestrecog/')
# print(df.shape)
# print(df.head())

X_train, X_test, y_train, y_test, enc_test= DataReader.read_Hand_Gest_DataFrame('./input/handGest/train/', './input/handGest/test/', Config.w, Config.h)
# train
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, batch_size=15, verbose=1, shuffle=True)
Model.saveAndPlot(history, Config.MODEL_FILE_NAME, model)

# load model
#model = tf.keras.models.load_model('handgest_model.h5')

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
        for i in range(Model.Config.final_class):
            if(maximum != max(maximum,inp[i])):
                maximum = max(maximum,inp[i])
                index = i
            inp[i] = 0
        inp[index]=1
    return y_pred
y_prediction  = binary_classify(y_prediction)


def create_result(y,enc):
    y_final = []
    for i in range(y.shape[0]):
        y_final.append(enc.inverse_transform(y[i].reshape(1, Config.final_class))[0][0])
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


y_class_result = create_result(y_prediction,enc_test)
y_class_desired = create_result(y_test,enc_test)
y_label_desired , y_label_result = label_encode(y_class_desired , y_class_result)

from sklearn.metrics import classification_report
tn = []
for cat in enc_test.categories_[0].reshape(Config.final_class,1):
    tn.append(cat[0])
target_names = tn
print(classification_report(y_label_desired, y_label_result, target_names=target_names))