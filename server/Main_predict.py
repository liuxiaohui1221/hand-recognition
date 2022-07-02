from PIL import Image
import base64
from sklearn import preprocessing
from model import Model
from model.Model import load_model, Config
from reader import DataReader
import tensorflow as tf
import cv2
import numpy as np
tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def binary_classify(y_pred,final_class=6):
    final_index=np.zeros(len(y_pred))
    pos=0
    for inp in y_pred:
        maximum = 0
        index = 0
        for i in range(final_class):
            if(maximum != max(maximum,inp[i])):
                maximum = max(maximum,inp[i])
                index = i
            inp[i] = 0
        inp[index]=1
        final_index[pos]=index
        pos+=1
    return y_pred,final_index

def get_base64_image(path):
    """open image and convert to base64"""
    with open(path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode()
    return image_base64
def base64_to_cv2(image_base64):
    """base64 image to cv2"""
    # print('imageBase64:',image_base64)
    image_bytes = base64.b64decode(image_base64)
    np_array = np.frombuffer(image_bytes, np.uint8)
    image_cv2 = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    return image_cv2

def predict(model,image_base64,single=True):
    X_test = DataReader.handle_img(base64_to_cv2(image_base64),Model.Config.w,Model.Config.h)
    y_prediction = model.predict(X_test)
    y_prediction, final_index = binary_classify(y_prediction)
    cls=final_index
    if single:
        cls=final_index[0]
    print("prediction num:", cls)
    return cls

model=load_model()
def testPredict(imgNum=2,suffix='.jpg'):

    # predict test
    # X_test = DataReader.handle_img(img)
    # X_test, y_test, enc_test= DataReader.readImageToDataFrame('1', './input/1.jpg')
    imgPath='E:/pycharm_workspace/hand-recognition/input/'+str(imgNum)+suffix
    final_index = predict(model,get_base64_image(imgPath))
    print("Actual: %s, predict: %s"%(imgNum,final_index))

testPredict(0)
testPredict(1,'.png')
testPredict(2)
testPredict(3,'.png')
testPredict(4)
testPredict(5,'.png')

def label_encode(y, y_pred):
    le = preprocessing.LabelEncoder()
    le.fit(y_pred)
    print(le.classes_)
    y = le.transform(y)
    y_pred = le.transform(y_pred)
    return y, y_pred

def create_result(y,enc):
    y_final = []
    for i in range(y.shape[0]):
        y_final.append(enc.inverse_transform(y[i].reshape(1, Config.final_class))[0][0])
    return y_final

X_train, X_test, y_train, y_test, enc_test= DataReader.read_Hand_Gest_DataFrame('../input/gen_img/train/', '../input/handGest/test/', Config.w, Config.h)
y_prediction = model.predict(X_test)
y_prediction,y_class_result  = binary_classify(y_prediction)
y_test,y_class_desired  = binary_classify(y_test)
y_label_desired , y_label_result = label_encode(y_class_desired , y_class_result)
print("实际标签：",y_label_desired)
print("预测标签：",y_label_result)
from sklearn.metrics import classification_report
tn = []
for cat in enc_test.categories_[0].reshape(Config.final_class,1):
    tn.append(cat[0])
target_names = tn
print(classification_report(y_label_desired, y_label_result, target_names=target_names))


