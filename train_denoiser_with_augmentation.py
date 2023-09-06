#%%
import urllib
import zipfile
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.metrics import RootMeanSquaredError

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
# val_input = np.load('data/Denoising_Data/val_input.npy')
# val_target = np.load('data/Denoising_Data/val_target.npy')

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
        x = tf.keras.layers.ReLU()(x)
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
# diameters = [112, 131, 151, 185, 200, 216, 292, 350]
diameters = [112, 131, 151, 185, 216, 292]

noise_files = [noise_patch_dir / f'diameter{d}mm.npy' for d in diameters]
noise_patch_dict = {f.stem: np.load(f) for f in noise_files}
noise_patches = np.concatenate(list(noise_patch_dict.values()))
# noise_patches = np.zeros_like(noise_patches)
# %%
aug_thresh = 0.3
def augment(image_label, seed, max_noise=1):
  image, label = image_label
  new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
  noise_patch = noise_patches[np.random.choice(list(range(len(noise_patches))))][:,:,None]
  noise_lambda = tf.random.uniform([1], minval=0, maxval=max_noise)[0]
  add_noise = tf.random.uniform([1], minval=0, maxval=1) > aug_thresh #from 0.5

  if add_noise:
    image = label + noise_lambda*noise_patch
  return image, label

# %%
example_input = test_input[[3], ...]
edge_buffer = 128
progress_ims = []
progress_val = []

class SaveSampleImageCallback(keras.callbacks.Callback):
    def __init__(self, progress_ims, progress_val):
          super().__init__()
          self.progress_ims = progress_ims
          self.progress_val = progress_val
    def on_epoch_begin(self, epoch, logs=None):
            val_loss = self.model.evaluate(val_input, val_target)
            example_output = self.model.predict(example_input)
            example_img = example_output[0, edge_buffer:-edge_buffer,
                                         edge_buffer:-edge_buffer, 0]
            self.progress_ims.append(example_img)
            self.progress_val.append(val_loss)

# %%
batch_size = 32
SHUFFLE_BUFFER_SIZE = 100
# %%
train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target))

val_input = noise_patch_dict['diameter131mm'][:1000,:,:,None]
val_target = np.zeros_like(val_input)
val_dataset = tf.data.Dataset.from_tensor_slices((val_input, val_target))

test_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_target))
# %%
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
# %%
from tensorflow.keras.metrics import RootMeanSquaredError
import datetime

rng = tf.random.Generator.from_seed(123, alg='philox')
def f(x, y):
  seed = rng.make_seeds(2)[0]
  image, label = augment((x, y), seed)
  return image, label

# This sets the number of iterations through the training data
epochs = 15
batch_size = 32
learning_rate = 0.0001
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
optimizer = 'adam'
# optimizer = tf.keras.optimizers.legacy.Adam(lr=learning_rate)

train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target))
val_dataset = tf.data.Dataset.from_tensor_slices((val_input, val_target))
test_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_target))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

AUTOTUNE = tf.data.AUTOTUNE
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def train(augment_training=False, model=None, epochs=15):
  'providing `model` as an argument skips model building and compiling and allows fine-tuning with a pretrained model'
  train_ds = (
      train_dataset
      .map(f, num_parallel_calls=AUTOTUNE)
      .prefetch(AUTOTUNE)
  ) if augment_training else (
      train_dataset
      .prefetch(AUTOTUNE)
  )
  if model is None:
    model = build_model()
    model.compile(optimizer=optimizer, loss='mse', metrics=[RootMeanSquaredError()])
  progress_ims = []
  progress_val = []
  callback = SaveSampleImageCallback(progress_ims, progress_val)
  model.fit(train_ds, epochs = epochs, callbacks=[callback, tensorboard_callback], validation_data=val_dataset)

  progress_ims = np.stack(progress_ims, axis=0)

  print('Training phase complete.')
  return model, np.stack(callback.progress_ims, axis=0), callback.progress_val

augment_training = False
print(f'Running augmented training: {augment_training}')
denoising_model, progress_ims, progress_val = train(augment_training=augment_training)

print("Evaluate on test set")
result = denoising_model.evaluate(test_dataset)
print(dict(zip(denoising_model.metrics_names, result)))

save_name = 'models/simple_cnn_denoiser'
tf.keras.models.save_model(denoising_model, save_name)

# now fine tune on pretrained model
augment_training = True
print(f'Running augmented training: {augment_training}')
denoising_model, progress_ims, progress_val = train(augment_training=augment_training, model=denoising_model, epochs=15)

print("Evaluate on test set")
result = denoising_model.evaluate(test_dataset)
print(dict(zip(denoising_model.metrics_names, result)))

save_name = 'models/simple_cnn_denoiser_augmented'
tf.keras.models.save_model(denoising_model, save_name)

# %%
