import numpy as np
import tensorflow as tf
from PIL import Image
from collections import OrderedDict
from tensorflow.contrib import predictor

#  加载模型,使用estimator导出的模型、tf.saved_model保存的模型都可以使用该方法
# 模型目录文件为：saved_model.pb  variables/xxx

predictor_fn = predictor.from_saved_model("./model/1602761609/")

feed_dict = OrderedDict()
img = np.array(Image.open('./data/test/.jpg').resize((224, 224))).astype(np.float) / 128 - 1

feed_dict['image'] = img

pred = predictor_fn(feed_dict)
print(pred)
