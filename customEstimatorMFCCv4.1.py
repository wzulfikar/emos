#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

#this version is the attempted adding function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils      

import argparse
import tensorflow as tf

import mfcc_data
from python_speech_features import delta
from python_speech_features import logfbank
import pyaudio
import wave
import sys
import numpy as np
import csv
import os.path

import emos

CHUNK = 1024
# What is CHUNKS here ?
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "03a01Wa.wav"

p = pyaudio.PyAudio()

def main(argv):

    # serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)
    

if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)