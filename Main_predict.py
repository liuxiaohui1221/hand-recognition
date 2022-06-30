from PIL import Image
import base64
from model.Model import load_model
from reader import DataReader
import tensorflow as tf
import cv2
import numpy as np
tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def binary_classify(y_pred,final_class=6):
    final_index=0
    for inp in y_pred:
        maximum = 0
        index = 0
        for i in range(final_class):
            if(maximum != max(maximum,inp[i])):
                maximum = max(maximum,inp[i])
                index = i
            inp[i] = 0
        inp[index]=1
        final_index=index
    return y_pred,final_index

def get_base64_image(path):
    """open image and convert to base64"""
    with open(path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode()
    return image_base64
def base64_to_cv2(image_base64):
    """base64 image to cv2"""
    print('imageBase64:',image_base64)
    image_bytes = base64.b64decode(image_base64)
    np_array = np.frombuffer(image_bytes, np.uint8)
    image_cv2 = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    return image_cv2

def predict(model,image_base64):
    X_test = DataReader.handle_img(base64_to_cv2(image_base64))
    y_prediction = model.predict(X_test)
    y_prediction, final_index = binary_classify(y_prediction)
    print("prediction num:", final_index)
    return final_index

def testPredict():

    # predict test
    # X_test = DataReader.handle_img(img)
    # X_test, y_test, enc_test= DataReader.readImageToDataFrame('1', './input/1.jpg')
    imgNum=2
    imgPath='E:/pycharm_workspace/hand-recognition/input/'+str(imgNum)+'.jpg'
    final_index = predict(load_model(),get_base64_image(imgPath))
    print("Actual: %s, predict: %s"%(imgNum,final_index))
testPredict()
