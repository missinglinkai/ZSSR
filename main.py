#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2
import os
import keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Model
from keras.layers import Conv2D, Input
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr
import glob
import missinglink

"""
Created on Wed Dec 19 15:37:28 2018
@author: shahar
"""

# scaling factor
SR_FACTOR = 2
# Activation layer
ACTIVATION = 'relu'
# Data generator random ordered
SHUFFLE = False
# scaling factors array order random or sorted
SORT = True
# Ascending or Descending: 'A' or 'D'
SORT_ORDER = 'A'
# number of time steps (pairs) per epoch
NB_STEPS = 1
# Batch size
BATCH_SIZE = 1
# Number of channels in signal
NB_CHANNELS = 3
# No. of NN filters per layer
FILTERS = 64  # 64 on the paper
# Number of internal convolutional layers
LAYERS_NUM = 6
# No. of scaling steps. 6 is best value from paper.
NB_SCALING_STEPS = 1
# No. of LR_HR pairs
EPOCHS = NB_PAIRS = 1500
# Default crop size (in the paper: 128*128*3)
CROP_SIZE = [96]#[32,64,96,128]
# Momentum # default is 0.9 # 0.86 seems to give lowest loss *tested from 0.85-0.95
BETA1 = 0.90  # 0.86
# Adaptive learning rate
INITIAL_LRATE = 0.001
DROP = 0.5
# Adaptive lrate, Number of learning rate steps (as in paper)
FIVE = 5
# Decide if learning rate should drop in cyclic periods.
LEARNING_RATE_CYCLES = False
#
# EPOCHS_DROP = np.ceil((NB_STEPS * EPOCHS ) / NB_SCALING_STEPS)
# Plot super resolution image when using zssr.predict
PLOT_FLAG = False
# Crop image for training
CROP_FLAG = True
# Flip flag
FLIP_FLAG = True
# initial scaling bias (org to fathers)
SCALING_BIAS = 1
# Scaling factors - blurring parameters
BLUR_LOW = 0.4
BLUR_HIGH = 0.95
# Add noise or not to transformations
NOISE_FLAG = False
# Mean pixel noise added to lr sons
NOISY_PIXELS_STD = 30
# Save augmentations
SAVE_AUG = True
# If there's a ground truth image. Add to parse.
GROUND_TRUTH = True
# If there's a baseline image. Add to parse.
BASELINE = True
# png compression ratio: best quality
CV_IMWRITE_PNG_COMPRESSION = 9

ORIGIN_IMAGE = 0
GROUND_TRUTH_IMAGE = 1
BASELINE_IMAGE = 2

# ---FUNCTIONS---#

""" Load an image."""
# TODO Delete change to number of channels. Check it and return it to main for the configuration of the model. Delete mean.
def load_img(file_name):
    # Load the image
    image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    print("Input image shape:", np.shape(image))
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.stack((image,) * 3, axis=-1)

    print(type(image))
    # Check and change image to channels last
    if image.shape[0] == 1 or image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)

    image = image.astype('float32')
    #    # Normalize img data
    #	image = image/255.0
    # image_mean = np.mean(image, dtype='float32')
    #	image = (image - image.mean()) / image.std()
#	plot_image(image)
    return image

# Add noise to lr sons
""" 0 is the mean of the normal distribution you are choosing from
    NOISY_PIXELS_STD is the standard deviation of the normal distribution
    (row,col,ch) is the number of elements you get in array noise   """
def add_noise(image):
    row, col, ch = image.shape

    noise = np.random.normal(0, NOISY_PIXELS_STD, (row, col, ch))
    #Check image dtype before adding.
    noise = noise.astype('float32')
    # We clip negative values and set them to zero and set values over 255 to it.
    noisy = np.clip((image + noise), 0, 255)

    return noisy


def preprocess(image, scale_fact, scale_fact_inter, i):

    # scale down is sthe inverse of the intermediate scaling factor
    scale_down = 1 / scale_fact_inter
    # Create hr father by downscaling from the original image
    hr = cv2.resize(image, None, fx=scale_fact, fy=scale_fact, interpolation=cv2.INTER_CUBIC)
    # Crop the HR father to reduce computation cost and set the training independent from image size
    #print("hr before crop:", hr.shape[0], hr.shape[1])
    if CROP_FLAG:
        h_crop = w_crop = np.random.choice(CROP_SIZE)
        #print("h_crop, w_crop:", h_crop, w_crop)
        if (hr.shape[0] > h_crop):
            x0 = np.random.randint(0, hr.shape[0] - h_crop)
            h = h_crop
        else:
            x0 = 0
            h = hr.shape[0]
        if (hr.shape[1] > w_crop):
            x1 = np.random.randint(0, hr.shape[1] - w_crop)
            w = w_crop
        else:
            x1 = 0
            w = hr.shape[1]
        hr = hr[x0 : x0 + h, x1 : x1 + w]
        #print("hr body:", x0, x0 + h, x1, x1 + w)
        #print("hr shape:", hr.shape[0], hr.shape[1])


    if FLIP_FLAG:
        # flip
        """ TODO check if 4 is correct or if 8 is better.
        Maybe change to np functions, as in predict permutations."""

        # if np.random.choice(4):
        #     flip_type = np.random.choice([1, 0, -1])
        #     hr = cv2.flip(hr, flip_type)
        #     if np.random.choice(2):
        #         hr = cv2.transpose(hr)
        k = np.random.choice(8)
        print(k)
        hr = np.rot90(hr, k, axes=(0, 1))
        if (k > 3):
            hr = np.fliplr(hr)

        # lr = cv2.flip( lr, flip_type )

    # hr is cropped and flipped then copies as lr
    # Blur lr son
    lr = cv2.resize(hr, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_CUBIC)
    # Upsample lr to the same size as hr
    lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Add gaussian noise to the downsampled lr
    if NOISE_FLAG:
        lr = add_noise(lr)
    # Save every Nth augmentation to artifacts.
    if SAVE_AUG and i%50==0:
        dir_name = os.path.join(output_paths, 'Aug/')
        # Create target Directory if don't exist
        mk_dir(dir_name=dir_name)

        cv2.imwrite(output_paths + '/Aug/' + str(SR_FACTOR) + '_' + str(i) + 'lr.png', cv2.cvtColor(lr, cv2.COLOR_RGB2BGR), params=[CV_IMWRITE_PNG_COMPRESSION])
        cv2.imwrite(output_paths + '/Aug/' + str(SR_FACTOR) + '_' + str(i) + 'hr.png', cv2.cvtColor(hr, cv2.COLOR_RGB2BGR), params=[CV_IMWRITE_PNG_COMPRESSION])

    # Expand image dimension to 4D Tensors.
    lr = np.expand_dims(lr, axis=0)
    hr = np.expand_dims(hr, axis=0)

    """ For readability. This is an important step to make sure we send the
    LR images as inputs and the HR images as targets to the NN"""
    X = lr
    y = hr

    return X, y


def s_fact(image, NB_PAIRS, NB_SCALING_STEPS):
    BLUR_LOW_BIAS = 0.0
    scale_factors = np.empty(0)
    if image.shape[0] * image.shape[1] <= 50 * 50:
        BLUR_LOW_BIAS = 0.3
    for i in range(NB_SCALING_STEPS):
        temp = np.random.uniform(BLUR_LOW + BLUR_LOW_BIAS, BLUR_HIGH,
                                 int(NB_PAIRS / NB_SCALING_STEPS))  # Low = 0.4, High = 0.95
        if SORT:
            temp = np.sort(temp)
        if SORT_ORDER == 'D':
            temp = temp[::-1]
        scale_factors = np.append(scale_factors, temp, axis=0)
        scale_factors = np.around(scale_factors, decimals=5)
    scale_factors_pad = np.repeat(scale_factors[-1], abs(NB_PAIRS - len(scale_factors)))
    scale_factors = np.concatenate((scale_factors, scale_factors_pad), axis=0)

    # Intermediate SR_Factors
    intermidiate_SR_Factors = np.delete(np.linspace(1, SR_FACTOR, NB_SCALING_STEPS + 1), 0)
    intermidiate_SR_Factors = np.around(intermidiate_SR_Factors, decimals=3)

    lenpad = np.int(NB_PAIRS / NB_SCALING_STEPS)
    intermidiate_SR_Factors = np.repeat(intermidiate_SR_Factors, lenpad)

    pad = np.repeat(intermidiate_SR_Factors[-1], abs(len(intermidiate_SR_Factors) - max(len(scale_factors), NB_PAIRS)))
    intermidiate_SR_Factors = np.concatenate((intermidiate_SR_Factors, pad), axis=0)

    #	scale_factors = np.vstack((scale_factors,a))
    return scale_factors, intermidiate_SR_Factors


def image_generator(image, NB_PAIRS, BATCH_SIZE, NB_SCALING_STEPS):
    i = 0
    scale_fact, scale_fact_inter = s_fact(image, NB_PAIRS, NB_SCALING_STEPS)
    while True:
        X, y = preprocess(image, scale_fact[i] + np.round(np.random.normal(0.0, 0.03), decimals=3),
                          scale_fact_inter[i], i)

        i = i + 1

        yield [X, y]


"""Our learning rate decay function is a periodic step function. It makes a step
every 'step_length' and resets itself back to the initial value every 'cycle'"""

def step_decay(epochs):
    initial_lrate = INITIAL_LRATE
    drop = DROP
    if LEARNING_RATE_CYCLES:
        cycle = np.ceil(NB_PAIRS / NB_SCALING_STEPS)
        epochs_drop = np.ceil((NB_STEPS * EPOCHS) / NB_SCALING_STEPS)
        step_length = int(epochs_drop / FIVE)
    else:
        cycle = NB_PAIRS
        epochs_drop = np.ceil((NB_STEPS * EPOCHS) / FIVE)
        step_length = epochs_drop

    lrate = initial_lrate * np.power(drop, np.floor((1 + np.mod(epochs, cycle)) / step_length))
    print("lrate", lrate)
    return lrate


def build_model():
    # model
    filters = FILTERS  # 64
    kernel_size = 3  # Highly important to keep image size the same through layer
    strides = 1  # Highly important to keep image size the same through layer
    padding = "same"  # Highly important to keep image size the same through layer
    inp = Input(shape=(None, None, NB_CHANNELS))

    # seq_model = Sequential()
    z = (Conv2D(
        filters=NB_CHANNELS,
        kernel_size=kernel_size,
        activation="relu",
        padding=padding,
        strides=strides,
        input_shape=(None, None, NB_CHANNELS)
    ))(inp)  # layer 1
    # Create inner Conv Layers
    for layer in range(LAYERS_NUM):
        z = (Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=ACTIVATION))(
            z)

    z = (Conv2D(filters=NB_CHANNELS, kernel_size=kernel_size, strides=strides, padding=padding, activation="linear"))(
        z)  # 8 - last layer - no relu

    # Residual layer
    out = keras.layers.add([z, inp])
    # FCN Model with residual connection
    zssr = Model(inputs=inp, outputs=out)
    # acc is not a good metric for this task*
    # compile model
    zssr.compile(loss='mae', optimizer='adam')
    # Model summary
    zssr.summary()
    # Plot model
    # from keras.utils import plot_model
    # plot_model(zssr, to_file = output_paths + 'zssr.png')
    return zssr


def ssim_calc(img1, img2, scaling_fact):
    #	img1 = cv2.resize(img1,None,fx=scaling_fact,fy=scaling_fact, interpolation=cv2.INTER_CUBIC)
    ssim_sk = ssim(img1, img2,
                   data_range=img1.max() - img1.min(), multichannel=True)
    print("SSIM:", ssim_sk)

    return ssim_sk


def psnr_calc(img1, img2, scaling_fact):
    # Get psnr measure from skimage lib
    PIXEL_MAX = 255.0
    PIXEL_MIN = 0.0
    sk_psnr = compare_psnr(img1, img2, data_range=PIXEL_MAX - PIXEL_MIN)
    print("PSNR:", sk_psnr)

    return sk_psnr


"""Check if a ground truth image exists, if so, compute metrics"""

def metric_results(ground_truth_image, super_image):
    try:
        ground_truth_image
        psnr_score = psnr_calc(ground_truth_image, super_image, SR_FACTOR)
        ssim_score = ssim_calc(ground_truth_image, super_image, SR_FACTOR)

    except NameError:
        psnr_score = None
        ssim_score = None

    return psnr_score, ssim_score


"This function creates a super resolution image and a simple interpolated image. returns and saves both of them."

def predict_func(image):
    # Resize original image to super res size
    interpolated_image = cv2.resize(image, None, fx=SR_FACTOR, fy=SR_FACTOR, interpolation=cv2.INTER_CUBIC)  # SR_FACTOR
    # Expand dims for NN
    interpolated_image = np.expand_dims(interpolated_image, axis=0)

    # Expand dims for NN
    # Check if image is a 4D tensor
    if len(np.shape(interpolated_image)) == 4:
        pass
    else:
        interpolated_image = np.expand_dims(interpolated_image, axis=0)
    # Get prediction from NN
    super_image = zssr.predict(interpolated_image)
    # Reduce the unwanted dimension and get image from tensor
    super_image = np.squeeze(super_image, axis=(0))
    interpolated_image = np.squeeze(interpolated_image, axis=(0))

    # Normalize data type back to uint8
    super_image = cv2.convertScaleAbs(super_image)
    interpolated_image = cv2.convertScaleAbs(interpolated_image)

    # Save super res image
    cv2.imwrite(output_paths + '/' + str(SR_FACTOR) + '_super.png', cv2.cvtColor(super_image, cv2.COLOR_RGB2BGR),
                params=[CV_IMWRITE_PNG_COMPRESSION])
    # Save bi-cubic enlarged image
    cv2.imwrite(output_paths + '/' + str(SR_FACTOR) + '_super_size_interpolated.png',
                cv2.cvtColor(interpolated_image, cv2.COLOR_RGB2BGR), params=[CV_IMWRITE_PNG_COMPRESSION])

    return super_image, interpolated_image


def accumulated_result(image):
    # Resize original image to super res size
    int_image = cv2.resize(image, None, fx=SR_FACTOR, fy=SR_FACTOR, interpolation=cv2.INTER_CUBIC)  # SR_FACTOR
    print("NN Input shape:", np.shape(int_image))
    super_image_accumulated = np.zeros(np.shape(int_image))
    # Get super res image from the NN's output
    super_image_list = []
    for k in range(0, 8):
        print("k", k)
        img = np.rot90(int_image, k, axes=(0, 1))
        if (k > 3):
            print("flip")
            img = np.fliplr(img)
        # Expand dims for NN
        img = np.expand_dims(img, axis=0)
        super_img = zssr.predict(img)
        super_img = np.squeeze(super_img, axis=(0))
        super_img = cv2.convertScaleAbs(super_img)
        # images should be UN-Rotated before added together
        # Make sure values are not threashold
        # First unflip and only then un-rotate to get the wanted result
        if (k > 3):
            print("unflip")
            super_img = np.fliplr(super_img)
        super_img = np.rot90(super_img, -k, axes=(0, 1))
        super_image_list.append(super_img)
        super_image_accumulated = super_image_accumulated + super_img

    super_image_accumulated_avg = np.divide(super_image_accumulated, 8)
    # Normalize data type back to uint8
    super_image_accumulated_avg = cv2.convertScaleAbs(super_image_accumulated_avg)
    cv2.imwrite(output_paths + '/' + str(SR_FACTOR) + '_super_image_accumulated_avg.png',
                cv2.cvtColor(super_image_accumulated_avg, cv2.COLOR_RGB2BGR), params=[CV_IMWRITE_PNG_COMPRESSION])

    super_image_accumulated_median = np.median(super_image_list, axis=0)
    ####	supsup = cv2.cvtColor(supsup, cv2.COLOR_RGB2BGR)
    super_image_accumulated_median = cv2.convertScaleAbs(super_image_accumulated_median)
    cv2.imwrite(output_paths + '/' + str(SR_FACTOR) + '_super_image_accumulated_median.png',
                cv2.cvtColor(super_image_accumulated_median, cv2.COLOR_RGB2BGR), params=[CV_IMWRITE_PNG_COMPRESSION])

    return super_image_accumulated_median, super_image_accumulated_avg


def select_first_dir(*dirs):
    d = None

    for d in dirs:
        if not os.path.isabs(d):
            d = os.path.join(os.path.dirname(__file__), d)

        if os.path.isdir(d):
            return d

    return d


def select_file(img_type,subdir):
    # save to disk
    if os.environ.get('ML'):
        path_input = '/data'
        path = os.path.join(path_input, '*.png')
    else:
        path_input = './ZSSR_Images'
        path = os.path.join(path_input, subdir, '*.png')

    # path = os.path.join(path_input, subdir, '*.png')
    # dir_path = select_first_dir('/data', path_subdir)

    # path = os.path.join(path_subdir, "*.png")

    print("path choice:",path)
    image_list = glob.glob(path)
    print("image_list:", image_list)
    # Choose Ground Truth image or Downsampled
    if img_type == 0:
        matching = [s for s in image_list if "LR" in s]
    elif img_type == 1:
        matching = [s for s in image_list if "HR" in s]
    else:  # img_type == 2
        matching = [s for s in image_list if "EDSR" in s]
    print("matching:", matching)

    try:
        image_path = str(matching[0])
    except IndexError:
        image_path = 'null'
    print("image_path:", image_path)

    return image_path


def mk_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory ", dir_name, " Created ")
    else:
        print("Directory ", dir_name, " already exists")
# main
if __name__ == '__main__':
    np.random.seed(0)
    if keras.backend == 'tensorflow':
        keras.backend.set_image_dim_ordering('tf')

    # Provide an alternative to provide MissingLinkAI credential
    parser = argparse.ArgumentParser()
    parser.add_argument('--srFactor', type=int)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    #parser.add_argument('--filepath', type=str)
    parser.add_argument('--subdir', type=str, default='067')
    parser.add_argument('--filters', type=int, default=FILTERS)
    parser.add_argument('--activation', default=ACTIVATION)
    parser.add_argument('--shuffle', default=SHUFFLE)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--layers', type=int, default=LAYERS_NUM)
    parser.add_argument('--sortOrder', default=SORT_ORDER)
    parser.add_argument('--scalingSteps', type=int, default=NB_SCALING_STEPS)
    parser.add_argument('--groundTruth', default=GROUND_TRUTH)
    parser.add_argument('--baseline', default=BASELINE)
    parser.add_argument('--flip', default=FLIP_FLAG)
    parser.add_argument('--noiseFlag', default=False)
    parser.add_argument('--noiseSTD', type = int, default=30)
    parser.add_argument('--project')
    # Override credential values if provided as arguments
    args = parser.parse_args()
    #file_name = args.filepath or file_name
    subdir = args.subdir
    SR_FACTOR = args.srFactor or SR_FACTOR
    FILTERS = args.filters or FILTERS
    EPOCHS = args.epochs or EPOCHS
    ACTIVATION = args.activation or ACTIVATION
    SHUFFLE = args.shuffle or SHUFFLE
    BATCH_SIZE = args.batch or BATCH_SIZE
    LAYERS_NUM = args.layers or LAYERS_NUM
    SORT_ORDER = args.sortOrder or SORT_ORDER
    NB_SCALING_STEPS = args.scalingSteps or NB_SCALING_STEPS
    GROUND_TRUTH = args.groundTruth or GROUND_TRUTH
    BASELINE = args.baseline or BASELINE
    FLIP_FLAG = args.flip or FLIP_FLAG
    NOISE_FLAG = args.noiseFlag or NOISE_FLAG
    NOISY_PIXELS_STD = args.noiseSTD or NOISY_PIXELS_STD
    # We're making sure These parameters are equal, in case of an update from the parser.
    NB_PAIRS = EPOCHS
    EPOCHS_DROP = np.ceil((NB_STEPS * EPOCHS) / NB_SCALING_STEPS)
    psnr_score = None
    ssim_score = None
    metrics_ratio = None

    # Path for Data and Output directories on Docker
    # save to disk
    if os.environ.get('ML'):
        output_paths = '/output'
    else:
        output_paths = os.path.join(os.getcwd(), 'output')
        if not os.path.exists(output_paths):
            os.makedirs(output_paths)
    print (output_paths)
    # output_paths = select_first_dir('/output', './output')
    # if (output_paths == './output'):
    #     mk_dir(dir_name='./output')

    file_name = select_file(ORIGIN_IMAGE, subdir)
    print(file_name)
    # Load image from data volumes
    image = load_img(file_name)
    cv2.imwrite(output_paths + '/'  + 'image.png',
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params=[CV_IMWRITE_PNG_COMPRESSION])

    # MissingLink callbacks
    missinglink_callback = missinglink.KerasCallback(project=args.project)
    missinglink_callback.set_properties(
        display_name='Keras neural network',
        description='2D fully convolutional neural network for single image super resolution')
    missinglink_callback.set_hyperparams(crop=CROP_SIZE,
                                         activation=ACTIVATION, sr_factor=SR_FACTOR,
                                         filters=FILTERS, noise_level_LR = NOISY_PIXELS_STD)

    # Build and compile model
    zssr = build_model()
    # Learning rate scheduler
    lrate = LearningRateScheduler(step_decay)
    # Callbacklist
    # checkpoint
    # filepath = "/output/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    #
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max')

    callbacksList = [lrate, missinglink_callback] #, checkpoint
    # TRAIN
    history = zssr.fit_generator(image_generator(image, NB_PAIRS, BATCH_SIZE, NB_SCALING_STEPS),
                                 steps_per_epoch=NB_STEPS, 
                                 epochs=EPOCHS, 
                                 shuffle=SHUFFLE, 
                                 callbacks=callbacksList
                                 max_queue_size=32,
                                 workers=16,
                                 use_multiprocessing=False,
                                 verbose=1)
    # Saving our model and weights
    zssr.save(output_paths + '/zssr_model.h5')
    # PREDICT
    # Get Super Resolution image by predicting on the original input with the trained NN
    super_image, interpolated_image = predict_func(image)
    # Get super resolution images
    super_image_accumulated_median, super_image_accumulated_avg = accumulated_result(image)

    # Compare to ground-truth if exists
    if GROUND_TRUTH:
        # Load Ground-Truth image if one exists
        try:
            file_path_ground_truth = select_file(GROUND_TRUTH_IMAGE, subdir)
            if os.path.isfile(file_path_ground_truth):
                ground_truth_image = load_img(file_path_ground_truth)
                cv2.imwrite(output_paths + '/' + '_ground_truth_image.png',
                        cv2.cvtColor(ground_truth_image, cv2.COLOR_RGB2BGR),
                        params=[CV_IMWRITE_PNG_COMPRESSION])

                print("Interpolated:")
                metric_results(ground_truth_image, interpolated_image)
                print("Super image:")
                psnr_score, ssim_score = metric_results(ground_truth_image, super_image)
                print("Super_image_accumulated_median")
                metric_results(ground_truth_image, super_image_accumulated_median)
                print("Super_image_accumulated_avg")
                metric_results(ground_truth_image, super_image_accumulated_avg)
            else:
                print("No Ground Truth Image")
        except:
            pass
                # In case we have a baseline for comparison such as EDSR we load the baseline-image and compare it with our SR-image
        if BASELINE:
            try:
                file_path_EDSR = select_file(BASELINE_IMAGE, subdir)
                if os.path.isfile(file_path_EDSR):
                    EDSR_image = load_img(file_path_EDSR)
                    print("EDSR")
                    cv2.imwrite(output_paths + '/' + '_ESDR.png',
                                cv2.cvtColor(EDSR_image, cv2.COLOR_RGB2BGR),
                                params=[CV_IMWRITE_PNG_COMPRESSION])
                    psnr_score_EDSR, _ = metric_results(ground_truth_image, EDSR_image)
                    if (psnr_score and psnr_score_EDSR) is not None:
                        metrics_ratio = psnr_score / psnr_score_EDSR
                        print("Metrics_ratio: ", metrics_ratio)
                else:
                    print("No Baseline Image")
            except:
                pass
        #  Setting up the experiment_id so we can later create external metrics
    experiment_id = missinglink_callback.experiment_id
    model_weights_hash = missinglink_callback.calculate_weights_hash(zssr)
    metrics = {'psnr': psnr_score, 'SSIM': ssim_score, 'ZSSR_EDSR_psnr_ratio': metrics_ratio}
    missinglink_callback.update_metrics(metrics, experiment_id=experiment_id)
