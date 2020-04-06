from numpy.random import seed
seed(8) #1
from tensorflow import set_random_seed
set_random_seed(7) #2

#

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model ,load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
from keras.applications import VGG16
from keras import optimizers




class TF_Model:

	def __init__(self):

		self.IMAGE_SIZE    = (150, 150)
		#self.NUM_CLASSES   = len(data_list)
		self.BATCH_SIZE    = 10  # try reducing batch size or freeze more layers if your GPU runs out of memory
		self.NUM_EPOCHS    = 20
		self.LEARNING_RATE =0.0005 #start off with high rate first 0.001 #5e-4

	def setup_dataset(self, DATASET_PATH):
		#Train datagen here is a preprocessor
		self.train_datagen = ImageDataGenerator(rescale=1./255,
										   rotation_range=50,
										   featurewise_center = True,
										   featurewise_std_normalization = True,
										   width_shift_range=0.2,
										   height_shift_range=0.2,
										   shear_range=0.25,
										   zoom_range=0.1,
										   zca_whitening = True,
										   channel_shift_range = 20,
										   horizontal_flip = True ,
										   vertical_flip = True ,
										   validation_split = 0.2,
										   fill_mode='constant')

		# test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
		#                                    fill_mode='constant')

		self.train_batches = self.train_datagen.flow_from_directory(DATASET_PATH,
														  target_size=self.IMAGE_SIZE,
														  shuffle=True,
														  batch_size=self.BATCH_SIZE,
														  subset = "training",
														  seed=42,
														  class_mode="binary",
														 
														  )

		self.valid_batches = self.train_datagen.flow_from_directory(DATASET_PATH,
														  target_size=self.IMAGE_SIZE,
														  shuffle=True,
														  batch_size=self.BATCH_SIZE,
														  subset = "validation",
														  seed=42,
														  class_mode="binary",
														  
														 
														  )





	def run(self, model_name, backbone, dataset_name):
		print("model name : "+ model_name)
		print("backbone : "+ backbone)
		print("dataset name : "+ dataset_name)
		print("training")

		dataaset_path = './data/' + dataset_name + '/content/two/train'
		self.setup_dataset(dataaset_path)
		
		conv_base = VGG16(weights='imagenet',
					  include_top=False,
					  input_shape=(150, 150, 3))


		conv_base.trainable = False


		model = models.Sequential()
		model.add(conv_base)
		model.add(layers.Flatten())
		model.add(layers.Dense(256, activation='relu'))
		model.add(layers.Dense(1, activation='softmax'))


		model.compile(loss='binary_crossentropy',
					  
					  optimizer=optimizers.Adam(lr=self.LEARNING_RATE),
					  metrics=['acc'])




		#FIT MODEL
		print(len(self.train_batches))
		print(len(self.valid_batches))

		STEP_SIZE_TRAIN=self.train_batches.n//self.train_batches.batch_size
		STEP_SIZE_VALID=self.valid_batches.n//self.valid_batches.batch_size

		result=model.fit_generator(self.train_batches,
								steps_per_epoch =STEP_SIZE_TRAIN,
								validation_data = self.valid_batches,
								validation_steps = STEP_SIZE_VALID,
								epochs= self.NUM_EPOCHS,                        
							   )
		model.save('Covid_Binary.h5')

