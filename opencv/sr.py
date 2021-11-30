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

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import cv2
import os
import numpy as np

from pycoral.adapters.common import input_size
from pycoral.adapters import common

from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    default_model_dir = '../srgan'
    default_model = 'srgan_quant_fix_edgetpu.tflite'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    
    args = parser.parse_args()

    
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    inference_size = input_size(interpreter)

    # get input scales
    scale_in, zero_point_in = common.input_details(interpreter, 'quantization')

    # get output scales
    output_details = interpreter.get_output_details()[0]
    # Outputs from the TFLite model are uint8, so we dequantize the results:
    scale_out, zero_point_out = output_details['quantization']
         
    

    cap = cv2.VideoCapture(args.camera_idx)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        

        cv2_inference = cv2.resize(frame, inference_size)
        cv2_im_rgb = cv2.cvtColor(cv2_inference, cv2.COLOR_BGR2RGB)

        # input preprocessing
        cv2_im_rgb = cv2.normalize(cv2_im_rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2_im_rgb = np.uint8(cv2_im_rgb / scale_in + zero_point_in)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        output = common.output_tensor(interpreter, 0)

        output = ((scale_out * (np.array(output).astype(np.float32) - zero_point_out))+1)*127.5
        sr = output.astype(np.uint8)[0]
        cv2_out = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        

        cv2.imshow('lr', cv2_inference)
        cv2.imshow('sr', cv2_out)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
