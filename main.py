# -*- coding: utf-8 from datetime import datetime

"""
Created on Sun Nov 25 23:47:26 2018

@author: shahar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 22:00:05 2018

@author: shahar
"""
import argparse
import numpy as np
import cv2
import os
import keras
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential, Model
from keras.layers import Conv2D, Input
#from configs import *
from keras import backend as K
#import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import skimage
from datetime import datetime
import errno

import missinglink
if K=='tensorflow':
    keras.backend.set_image_dim_ordering('tf')

pwd = '/home/shahar/mlinkdev/ZSSR2D/ZSSR_Mlink'
os.chdir(pwd)

# Mlink
MlinkFlag = True
# missinglink integration data
OWNER_ID = '8860d02a-12cb-4a88-870a-3782469bffc0'
PROJECT_TOKEN = 'rnjmAvqzsdJPEMuK'

#Activation layer
ACTIVATION = 'relu'
# Data generator random ordered
SHUFFLE = SORT = True

# No of network runs - outputs to collect and take the meadian
N_IMAGES=3
# number of time steps (pairs) per epoch
NB_STEPS = 1    ####None is changed to 1
# Batch size
BATCH_SIZE = 1
# Number of cahnnels in signal
NB_CHANNELS = 3 ##### FIXED FROM 1 TO 3
# No. of NN filters per layer
FILTERS = 64 #64 on the paper
# No. of scaling steps for training
NB_SCALING_STEPS = 1
# No. of LR_HR pairs
EPOCHS = NB_PAIRS = 100
# Default crop size (in the paper: 128*128*3)
#CROP_SIZE = 50000
# Momentum # default is 0.9 # 0.86 seems to give lowest loss *tested from 0.85-0.95
BETA1 = 0.90 #0.86
# Adaptive learning rate
INITIAL_LRATE = 0.001
DROP = 0.5
# Early stop
PATIENCE = 30
#EPOCHS = 1000 #350 #best for small nb_pairs
EPOCHS_DROP = int((NB_STEPS * EPOCHS * NB_SCALING_STEPS) / 5)
# Plot super resolution image when using zssr.predict
PLOT_FLAG = False
# Crop image for training
CROP_FLAG = True
# scaling factor
SR_FACTOR = 4
# Add noise or not to transformations
NOISE_FLAG = False
# Flip flag
FLIP_FLAG = False
# Add noise during pair generation
NOISE_FLAG_GEN = True
# No. of scaling steps. 6 is best value from paper.
NB_SCALING_STEPS = 1
# initial scaling bias (org to fathers)
SCALING_BIAS = 1
# Maximum size of pool matrix
POOL_SIZE = [NB_PAIRS]
# png compression ratio: best quality
CV_IMWRITE_PNG_COMPRESSION = 9



#MAIN
np.random.seed(0)

def dircreation():#list, filename):
	mydir = os.path.join(
	os.getcwd() + '/results',
		'Exp')#_' + datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
	try:
		os.makedirs(mydir)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise  # This was not a "directory exist" error..

	return mydir

dir_path = dircreation() + '/'


file_name =  os.getcwd() + '/Images/lincoln.png'
parser = argparse.ArgumentParser()
parser.add_argument('--filepath')

# Override credential values if provided as arguments
args = parser.parse_args()
file_name = args.filepath or file_name

# model
filters = FILTERS #64
kernel_size = 3
strides = 1
padding = "same"

seq_model = Sequential()
seq_model.add(Conv2D(
	filters=NB_CHANNELS,
	kernel_size=kernel_size,
	activation="relu",
	padding=padding,
	strides=strides,
	input_shape=(None, None, NB_CHANNELS)
	))  # layer 1

seq_model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=ACTIVATION)) #2
seq_model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=ACTIVATION)) #3
seq_model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=ACTIVATION)) #4
seq_model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=ACTIVATION)) #5
seq_model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=ACTIVATION)) #6
seq_model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=ACTIVATION)) #7
seq_model.add(Conv2D(filters=NB_CHANNELS, kernel_size=kernel_size, strides=strides, padding=padding, activation="linear")) #8 - last layer - no relu

#seq_model.summary()
#seq_model.outputs

#define model inputs
inputs = Input(shape=(None, None, NB_CHANNELS))
#x is modified signal after FCN
z = seq_model(inputs)
#residual signal = modified signal after FCN + original input signal
output = keras.layers.add([z, inputs])
# FCN Model with residual connection
zssr = Model(inputs, output)
#acc is not a good metric for this task*
#compile model
zssr.compile(loss='mae', optimizer='SGD')#, metrics = ['mean_squared_error'])#, psnr_run]) #'accuracy', 'hinge'
# Model summary
zssr.summary()
# Plot Model
from keras.utils import plot_model
plot_model(zssr, to_file= dir_path + 'zssr.png')


# Load image

def load_img(file_name):
    # Load the image
    # plt way
	 #'1158.jpg', , 'lincoln.png','img_004_SRF_2_LR.jpg'. '58udw75.jpg','img_034_SRF_2_LR.png'
	image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	print (type(image))
	image = image.astype('float32')
#    # Normalize img data
	image_mean = np.mean(image,dtype='float32')
#	image = (image - image.mean()) / image.std()
#	plot_image(image)
	return image, image_mean

image, image_mean =  load_img(file_name)



# TRAIN NN


def add_noise(image):
	row,col,ch= image.shape
	mean = np.random.normal(0,0.08)
	var = np.random.normal(0.1,0.001)
	sigma = var**0.5
	gauss = np.random.normal(mean,sigma,(row,col,ch))
	gauss = gauss.reshape(row,col,ch)
	gauss = gauss.astype('float32')
	noisy = image + gauss

	return noisy

def preprocess(image, scale_factor, SR_FACTOR):
#Rescale Image Rotate Image Resize Image Flip Image PCA etc.
	print("scale_factor:",scale_factor)
	scale_down = 1/SR_FACTOR
	hr = cv2.resize(image,None,fx=scale_factor,fy=scale_factor, interpolation=cv2.INTER_CUBIC)
#	cv2.imwrite(str(eval('scale_factor')) +'HR.png', cv2.cvtColor(hr, cv2.COLOR_RGB2BGR),params = [CV_IMWRITE_PNG_COMPRESSION ] )
	hr_shape = hr.shape
#	print("shape1",hr.shape)
	lr =  cv2.resize(hr,None,fx=scale_down,fy=scale_down, interpolation=cv2.INTER_CUBIC)
#	cv2.imwrite(str(eval('scale_factor')) +'LR_D.png', cv2.cvtColor(lr, cv2.COLOR_RGB2BGR),params = [CV_IMWRITE_PNG_COMPRESSION ] )
#	print("shape2",lr.shape)
	# Upsample lr to the same size as hr
	lr = cv2.resize(lr,(hr_shape[1],hr_shape[0]), interpolation=cv2.INTER_CUBIC)
	# Add gaussian noise to the downsampled lr
	lr = add_noise(lr)
#	cv2.imwrite(str(eval('scale_factor')) +'LR_U.png', cv2.cvtColor(lr, cv2.COLOR_RGB2BGR),params = [CV_IMWRITE_PNG_COMPRESSION ] )
#	print("shape3",lr.shape)
	if CROP_FLAG:
		x0 = x1 = np.random.randint(0,np.int(lr.shape[0]/2))
		h = w = np.random.randint((np.int(lr.shape[0] / 2)),(np.int(lr.shape[0] - 1)))
		lr = lr[x1:x1+h, x0:x0+w]
		hr = hr[x1:x1+h, x0:x0+w]

	lr = np.expand_dims(lr, axis=0)
	hr = np.expand_dims(hr, axis=0)

	X = lr
	y = hr

	return X, y

def s_fact():

	scale_factors = np.random.uniform(0.4,0.95,NB_PAIRS)
	scale_factors = np.around(scale_factors, decimals=5)
	if SORT: scale_factors.sort()
	# Intermidiate SR_Factors
	intermidiate_SR_Factors = np.delete(np.linspace(1,SR_FACTOR,NB_SCALING_STEPS + 1),0)
	intermidiate_SR_Factors = np.around(intermidiate_SR_Factors, decimals = 5)

	return scale_factors


def image_generator(file_name, NB_PAIRS, NB_STEPS):

	i=0
	image, image_mean = load_img(file_name)
	scale_factors = s_fact()
	while True:

#		for scale_factor in scale_factors:
#		print("scale_factor_img_gen_loop:",scale_factors[i])
		X, y = preprocess(image, scale_factors[i] + np.random.normal(0.0,0.01), SR_FACTOR)
		i=i+1

		yield [X, y]


#model_hist = train(zssr, file_name)
def step_decay(epoch):

    initial_lrate = INITIAL_LRATE
    drop = DROP
    epochs_drop = EPOCHS_DROP
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))

    return lrate

# Learning rate schedualer
lrate = LearningRateScheduler(step_decay)

# missinglink callbacks
missinglink_callback = missinglink.KerasCallback(owner_id=OWNER_ID, project_token=PROJECT_TOKEN)
missinglink_callback.set_properties(
	    	display_name='Keras neural network',
	        description='Two dimensional fully convolutional neural network for super resolution')
missinglink_callback.set_hyperparams(
			   activation=ACTIVATION, shuffle = SHUFFLE)


callbacksList = [missinglink_callback, lrate]
history = zssr.fit_generator(image_generator(file_name, NB_PAIRS, BATCH_SIZE),
						 steps_per_epoch=NB_STEPS, epochs=EPOCHS, shuffle = SHUFFLE, callbacks = callbacksList)




# PREDICT
#Get Super Resolution image by predicting on the original input with the trained NN

# Resize original image to super res size
interpolated_image = cv2.resize(image,None,fx=SR_FACTOR,fy=SR_FACTOR, interpolation=cv2.INTER_CUBIC) # SR_FACTOR
#	image = image.astype('uint8')
#super_image = super_image.astype('uint8')
# Save enlarged image
cv2.imwrite( dir_path + str(eval('SR_FACTOR')) + '_super_size_interpolated.png', cv2.cvtColor(interpolated_image, cv2.COLOR_RGB2BGR),params = [CV_IMWRITE_PNG_COMPRESSION ] )
# Expand dims for NN
interpolated_image = np.expand_dims(interpolated_image, axis=0)
# Get super res image from the NN's output
super_image = zssr.predict(interpolated_image)
#super_image = zssr.predict_generator(image_generator, steps=NB_STEPS, workers=1, use_multiprocessing=False, verbose=0)

# Get rid of extra dims
super_image = np.squeeze(super_image,axis=(0))
interpolated_image = np.squeeze(interpolated_image,axis=(0))
# Normalize data type back to uint8
#super_image = super_image.astype('uint8')
super_image = cv2.convertScaleAbs(super_image)
interpolated_image = cv2.convertScaleAbs(interpolated_image)
# Save super res image
cv2.imwrite( dir_path + str(eval('SR_FACTOR')) + '_super.png', cv2.cvtColor(super_image, cv2.COLOR_RGB2BGR),params = [CV_IMWRITE_PNG_COMPRESSION ] )
#	plot_image(super_image)

def ssim_calc(img1, img2, scaling_fact):

#	img1 = cv2.resize(img1,None,fx=scaling_fact,fy=scaling_fact, interpolation=cv2.INTER_CUBIC)
	ssim_sk = ssim(img1, img2,
                  data_range=img1.max() - img1.min(), multichannel=True)
	print("SSIM:", ssim_sk)

def psnr(img1, img2, scaling_fact):

    # Resize original image to super res size
#    img1 = cv2.resize(img1,None,fx=scaling_fact,fy=scaling_fact, interpolation=cv2.INTER_CUBIC)
    # Get psnr measure from skimage lib
    sk_psnr = skimage.measure.compare_psnr(img1, img2, data_range=max(img1.max() - img1.min(),img2.max() - img2.min()))
    # Calculate mse of original image vs. super res image
    mse = np.mean((img1 - img2)**2)
    # If there's no difference return 100
    if mse == 0: return 100, sk_psnr
    # Calculate psnr
    PIXEL_MAX = 255.0
    # Changed form 20 to 10 to be equal to skleanrs' method
    d = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    print ("PSNR:", d, sk_psnr)

psnr(interpolated_image, super_image, SR_FACTOR)
ssim_calc(interpolated_image, super_image, SR_FACTOR)
