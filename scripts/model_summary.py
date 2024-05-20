import os
import sys
from datetime import datetime

# Imports the root directory to the path in order to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf
from tensorflow import keras

import modules.model_architectures as model_architectures
from modules.pipeline import Pipeline
from modules.custom_learning_rate import CustomLearningRateScheduler

model = model_architectures.UNET_model_skip((512, 512), 5)

model.summary()
