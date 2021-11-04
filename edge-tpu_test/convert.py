# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to work with detection models."""

import numpy as np






def set_input_tensor(interpreter, image):

  input_details = interpreter.get_input_details()[0]
  tensor_index = input_details['index']
  
  data = np.asarray(image)
  shape = np.expand_dims(data, 0)
  print(shape.shape)
  print(input_details)
  interpreter.resize_tensor_input(tensor_index, shape.shape);         # change input size dynamically
  interpreter.allocate_tensors()
  
  input_details = interpreter.get_input_details()[0]
  input_tensor = interpreter.tensor(tensor_index)()[0]
  # Inputs for the TFLite model must be uint8, so we quantize our input data.
  # NOTE: This step is necessary only because we're receiving input data from
  # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
  # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
  #input_tensor[:, :] = input
  print(input_tensor.shape)
  scale, zero_point = input_details['quantization']
  
  data = np.float32(data)/255.0
  input_tensor[:, :] = np.uint8(data / scale + zero_point)




def output_tensor(interpreter, i):
  """Returns output tensor view."""
  tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
  return np.squeeze(tensor)


def get_output(interpreter):
  """Returns list of detected objects."""
  output_details = interpreter.get_output_details()[0]
  output = interpreter.get_tensor(output_details['index'])[0]
  
  # Outputs from the TFLite model are uint8, so we dequantize the results:
  scale, zero_point = output_details['quantization']
  output = ((scale * (output.numpy().astype(np.float32) - zero_point))+1)*127.5
  
  output = output.astype(np.uint8)
  # print(output)
  return output
