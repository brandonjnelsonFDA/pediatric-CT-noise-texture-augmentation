#%% -*- coding: utf-8 -*-
import urllib
import zipfile
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# %%
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if not os.path.exists('data'): os.mkdir('data')
if not os.path.exists('data/Denoising_Data'):
  urllib.request.urlretrieve('https://docs.google.com/uc?export=download&id=1-ZqL_1cqWeG6LsRAB0TwiddW8TgQ-q70', 'data/Denoising_Data.zip')
  with zipfile.ZipFile('data/Denoising_Data.zip', 'r') as zip_ref:
     zip_ref.extractall('data')
# %%
# load training data, used to update CNN weights
# 2000 30x30 image patches from 8 patients
train_input = np.load('data/Denoising_Data/train_input.npy')
train_target = np.load('data/Denoising_Data/train_target.npy')
# load validation data, used to monitor for overfitting
# 1000 30x30 image patches from 1 patient
val_input = np.load('data/Denoising_Data/val_input.npy')
val_target = np.load('data/Denoising_Data/val_target.npy')

# load testing data, used for evaluating performance
# 5 512x512 images from 1 patient
test_input = np.load('data/Denoising_Data/test_input.npy')
test_target = np.load('data/Denoising_Data/test_target.npy')

# Load examples images from state-of-the-art CNN denoising for CT images
test_example = np.load('data/Denoising_Data/test_input_denoised.npy')

print('Data loading completed.')

# %%
n_layers = 6
filters = 64
kernel_size = (3, 3)
strides = (1, 1)
activation = 'relu'

def build_model():
    xin = keras.layers.Input(shape=(None, None, 1), name='input_CT_images')
    # We define a preprocessing layer to rescale the CT image pixel values
    shift_mean = train_input.mean()
    rescale = train_input.std()
    x = keras.layers.Lambda(
        lambda x: (x - shift_mean) / rescale,
        name='normalization')(xin)

    for i in range(n_layers - 1):
        x = keras.layers.Conv2D(
          filters=filters,
          kernel_size=kernel_size,
          strides=strides,
          padding='same')(x)
        x = keras.layers.Activation(activation)(x)
    # Final layer has just one feature map corresponding to the output image
    x = keras.layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=strides,
        padding='same')(x)
    # Here we rescale the output to typical CT number range
    xout = keras.layers.Lambda(
        lambda x: (x * rescale) + shift_mean,
        name='output_CT_images')(x)
    # We define the model by specifying the inputand output tensors
    model = keras.Model(inputs=xin, outputs=xout, name="CT_denoiser")
    return model

noise_patch_dir = Path('noise_patches')
noise_patch_dict = {f.stem: np.load(f) for f in noise_patch_dir.glob('*.npy')}
noise_patches = np.concatenate(list(noise_patch_dict.values()))

def augment(image_label, seed, max_noise=1):
  image, label = image_label
  new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
  noise_patch = noise_patches[np.random.choice(list(range(len(noise_patches))))][:,:,None]
  noise_lambda = tf.random.uniform([1], minval=0, maxval=max_noise)

  add_noise = tf.random.uniform([1], minval=0, maxval=1) > 0.5
  if add_noise:
    image = label + noise_lambda[0]*noise_patch
    # image = image + noise_lambda[0]*noise_patch
  return image, label

# %%
batch_size = 32
SHUFFLE_BUFFER_SIZE = 100
train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target))
val_dataset = tf.data.Dataset.from_tensor_slices((val_input, val_target))
test_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_target))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

rng = tf.random.Generator.from_seed(123, alg='philox')
def f(x, y):
  seed = rng.make_seeds(2)[0]
  image, label = augment((x, y), seed)
  return image, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target))
val_dataset = tf.data.Dataset.from_tensor_slices((val_input, val_target))
test_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_target))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

denoising_model = build_model()
learning_rate = 0.0001
optimizer = tf.keras.optimizers.legacy.Adam(lr=learning_rate)
denoising_model.compile(optimizer=optimizer, loss='mse')

# This sets the number of iterations through the training data
epochs = 15
batch_size = 32
learning_rate = 0.0001
optimizer = tf.keras.optimizers.legacy.Adam(lr=learning_rate)

progress_example = 2
buffer = 128
progress_ims = []

AUTOTUNE = tf.data.AUTOTUNE

def train(loss_function, model, augment_training=False):

    train_ds = (
        train_dataset
        .map(f, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    ) if augment_training else (
        train_dataset
       .prefetch(AUTOTUNE)
    )
    model.compile(optimizer=optimizer, loss=loss_function)

    # As the training progresses, we'll monitor network output and performance
    # metrics. Some related variables are initialized here
    example_input = test_input[[3], ...]
    edge_buffer = 128
    progress_ims = []
    progress_val = []

    for epoch in range(epochs):
      # Evaluate model on reserved data
      val_loss = model.evaluate(val_input, val_target)
      example_output = model.predict(example_input)
      example_img = example_output[0, edge_buffer:-edge_buffer,
                                edge_buffer:-edge_buffer, 0]
      progress_ims.append(example_img)
      progress_val.append(val_loss)
      # Update model weights using training data
      istart = 0
      while istart < (len(train_input) - batch_size):
        for x, y in train_ds.take(1):
            model.train_on_batch(x=x, y=y)
        istart += batch_size

    progress_ims = np.stack(progress_ims, axis=0)

    print('Training phase complete.')
    return model, progress_ims, progress_val

for augment_training in [True, False]:
  denoising_model = build_model()
  denoising_model.compile(optimizer=optimizer, loss='mse')
  print(f'Running augmented training: {augment_training}')
  denoising_model, progress_ims, progress_val = train(loss_function='mse', model=denoising_model, augment_training=augment_training)

  save_name = 'models/simple_cnn_denoiser_augmented' if augment_training else 'models/simple_cnn_denoiser'
  tf.keras.models.save_model(denoising_model, save_name)
