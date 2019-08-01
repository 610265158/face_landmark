import tensorflow as tf





##make sure set bn in the right mode
graph_def_file = "./model/loss.pb"
input_arrays = ["tower_0/images"]
output_arrays = ["tower_0/prediction"]

input_shapes={"tower_0/images" : [1, 160, 160, 3]}
converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays,input_shapes=input_shapes)
tflite_model = converter.convert()
open("./model/loss.tflite", "wb").write(tflite_model)