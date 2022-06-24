from reader import DataReader
import tensorflow as tf

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

# load model
model = tf.keras.models.load_model('handgest_model.h5')

# predict
X_test, y_test, enc_test= DataReader.readImageToDataFrame('2', './input/2.jpg')
y_prediction = model.predict(X_test)
y_prediction,final_index  = binary_classify(y_prediction)

print("prediction num:",final_index)
