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
"""Example using TF Lite to detect objects in a given image."""

import argparse
import time

from PIL import Image
from PIL import ImageDraw

import convert
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]





def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])





def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-i', '--input', required=True,
                      help='File path of image to process.')
  parser.add_argument('-o', '--output',
                      help='File path for the result image with annotations')
 
  args = parser.parse_args()

  image = Image.open(args.input)


  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  
  convert.set_input_tensor(interpreter, image)


  print('----CONVERT TIME----')
  print('Note: The first inference is slow because it includes',
        'loading the model into Edge TPU memory.')
  
  start = time.perf_counter()
  interpreter.invoke()
  inference_time = time.perf_counter() - start
  sr = convert.get_output(interpreter)
  print('%.2f ms' % (inference_time * 1000))

  
  if args.output:
    sr = Image.fromarray(sr)
    sr.save(args.output)
    sr.show()


if __name__ == '__main__':
  main()