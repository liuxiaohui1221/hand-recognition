import os
import random

import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

#### 植物病害数据集 ######
# 调整亮度.对比度.饱和度.色相的顺序可以得到不同的结果
# 预处理时随机选择的一种,降低无关因素对模型的影响
# 原文链接：https://blog.csdn.net/akadiao/article/details/78547474
def distort_color(image, color_ordering=0, direction=0):
	##adjust direction
	if direction==1:
		image = tf.image.flip_up_down(image)
	elif direction==2:
		image = tf.image.flip_left_right(image)
	# adjust color
	if color_ordering == 0:
		image = tf.image.random_brightness(image, max_delta=32. / 255)
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.random_hue(image, max_delta=0.2)
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
	elif color_ordering == 1:
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		image = tf.image.random_brightness(image, max_delta=32. / 255)
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.random_hue(image, max_delta=0.2)
	elif color_ordering == 2:
		image = tf.image.random_brightness(image, max_delta=32. / 255)
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.random_hue(image, max_delta=0.2)
	elif color_ordering == 3:
		image = tf.image.random_brightness(image, max_delta=32. / 255)
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		image = tf.image.random_hue(image, max_delta=0.2)
	elif color_ordering == 4:
		image = tf.image.random_brightness(image, max_delta=32. / 255)
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		image = tf.image.random_hue(image, max_delta=0.2)
	elif color_ordering == 5:
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.random_brightness(image, max_delta=32. / 255)
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		image = tf.image.random_hue(image, max_delta=0.2)

	# 这一步转换张量数据类型很重要
	img = tf.cast(image, dtype=tf.uint8)
	image = tf.image.encode_png(img)
	return image


def img_gen(img_path, img_name, out_img_base,gen_nums=5):
	if not os.path.exists(out_img_base):
		os.makedirs(out_img_base)
	# img_path = "C:/Downloads/test/1.jpg"
	# 读入图片
	img = tf.io.read_file(img_path)
	#print(img)
	# image = Image.open(img_path)
	# plt.imshow(image)
	# plt.show()
	##img = image.resize([64,64])

	img = tf.image.decode_png(img, channels=3)
	# image_decode_jpeg = tf.image.decode_image(img)
	# image_decode_jpeg = tf.image.convert_image_dtype(img, dtype=tf.float32)

	##图像转换
	for i in range(gen_nums):
		image_new = distort_color(img,color_ordering=i,direction=random.randint(0,3))
		image_new_path = os.path.join(out_img_base, str(i)+"_"+img_name)
		#print(image_new_path)
		hd = tf.io.gfile.GFile(image_new_path, "w")
		hd.write(image_new.numpy())
		hd.close()
	#print("test end!")

##test
img_gen("C:/Downloads/test/1.JPG", "1.JPG", "C:/Downloads/test")

# 如果目录名字为中文 需要转码处理
# 图像路径：图像所在文件夹的父目录：比如'E:\\data'，该目录下的图像存在于0,1,...5个子目录中
dataBase = 'E:\\data'
outBase = dataBase
# 生成个数
gen_Nums=10
class_path=[]
img_path=[]
for className in os.listdir(dataBase) :
	class_path.append(className)
	#print(fileName)
for path in class_path:
	classPath=os.path.join(dataBase,path)
	outClassPath=os.path.join(outBase,path)
	#print(classPath)
	for fileName in os.listdir(classPath):
		#print(os.path.join(classPath,fileName))
		#img_path.append(os.path.join(classPath,fileName))
		img_gen(os.path.join(classPath,fileName), fileName, outClassPath,gen_nums=gen_Nums)
		pass
	print("End class:",classPath)
print("End all!")

