import tensorflow as tf
import numpy as np
import time

saved_model_dir='./model/keypoints'

save_tf_model="converted_model.tflite"


converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

##write it down
open(save_tf_model, "wb").write(tflite_model)

# 加载 TFLite 模型并分配张量（tensor）。
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# 获取输入和输出张量。
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 使用随机数据作为输入测试 TensorFlow Lite 模型。
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

tflite_results = interpreter.get_tensor(output_details[2]['index'])
start=time.time()

for i in range(100):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[2]['index'])

print('time cost:',(time.time()-start)/100.)
print('tflite result')
print(tflite_results)





model=tf.saved_model.load(saved_model_dir)
tf_results = model.inference(tf.constant(input_data))
print('tf result')
print(tf_results['landmark'])
