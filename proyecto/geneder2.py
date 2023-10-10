import os
import random
import shutil
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt


source_path = './gender_dataset_face'

source_path_man = os.path.join(source_path, 'man')
source_path_woman = os.path.join(source_path, 'woman')

print(f"There are {len(os.listdir(source_path_man))} images of man.")
print(f"There are {len(os.listdir(source_path_woman))} images of woman.")

root_dir = './man-vs-woman/'

if os.path.exists(root_dir):
  shutil.rmtree(root_dir)
  
def create_train_val_dirs(root_path):
  os.makedirs(os.path.join(root_path, 'training'))
  os.makedirs(os.path.join(f'{root_path}/training', 'man'))
  os.makedirs(os.path.join(f'{root_path}/training', 'woman'))
  os.makedirs(os.path.join(root_path, 'validation'))
  os.makedirs(os.path.join(f'{root_path}/validation', 'man'))
  os.makedirs(os.path.join(f'{root_path}/validation', 'woman'))
  
try:
  create_train_val_dirs(root_path=root_dir)
except FileExistsError:
  print("You should not be seeing this since the upper directory is removed beforehand")
  
for rootdir, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        print(os.path.join(rootdir, subdir))


def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
  dir_content = random.sample(os.listdir(SOURCE_DIR), len(os.listdir(SOURCE_DIR)))
  train_size = int(len(dir_content) * SPLIT_SIZE)
  for i, image_name in enumerate(dir_content):
        if os.path.getsize(os.path.join(SOURCE_DIR, image_name)) > 0:
            if i < train_size:
                copyfile(os.path.join(SOURCE_DIR, image_name), os.path.join(TRAINING_DIR, image_name))
            else:
                copyfile(os.path.join(SOURCE_DIR, image_name), os.path.join(VALIDATION_DIR, image_name))
WOMAN_SOURCE_DIR = "./gender_dataset_face/woman"
MEN_SOURCE_DIR = "./gender_dataset_face/man"

TRAINING_DIR = "./man-vs-woman/training"
VALIDATION_DIR = "./man-vs-woman/validation"

TRAINING_WOMAN_DIR = os.path.join(TRAINING_DIR, "woman/")
VALIDATION_WOMAN_DIR = os.path.join(VALIDATION_DIR, "woman/")

TRAINING_MEN_DIR = os.path.join(TRAINING_DIR, "man/")
VALIDATION_MEN_DIR = os.path.join(VALIDATION_DIR, "man/")

if len(os.listdir(TRAINING_WOMAN_DIR)) > 0:
    for file in os.scandir(TRAINING_WOMAN_DIR):
        os.remove(file.path)
        
if len(os.listdir(TRAINING_MEN_DIR)) > 0:
    for file in os.scandir(TRAINING_MEN_DIR):
        os.remove(file.path) 
        
if len(os.listdir(VALIDATION_WOMAN_DIR)) > 0:
    for file in os.scandir(VALIDATION_WOMAN_DIR):
        os.remove(file.path)
        
if len(os.listdir(VALIDATION_MEN_DIR)) > 0:
    for file in os.scandir(VALIDATION_MEN_DIR):
        os.remove(file.path)

split_size = .9

split_data(WOMAN_SOURCE_DIR, TRAINING_WOMAN_DIR, VALIDATION_WOMAN_DIR, split_size)
split_data(MEN_SOURCE_DIR, TRAINING_MEN_DIR, VALIDATION_MEN_DIR, split_size)

print(f"\n\nOriginal cat's directory has {len(os.listdir(WOMAN_SOURCE_DIR))} images")
print(f"Original dog's directory has {len(os.listdir(MEN_SOURCE_DIR))} images\n")

print(f"There are {len(os.listdir(TRAINING_WOMAN_DIR))} images of woman for training")
print(f"There are {len(os.listdir(TRAINING_MEN_DIR))} images of man for training")
print(f"There are {len(os.listdir(VALIDATION_WOMAN_DIR))} images of women for validation")
print(f"There are {len(os.listdir(VALIDATION_MEN_DIR))} images of man for validation")

def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
  """
  Creates the training and validation data generators

  Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images

  Returns:
    train_generator, validation_generator - tuple containing the generators
  """
  ### START CODE HERE

  # Instantiate the ImageDataGenerator class (don't forget to set the arguments to augment the images)
  train_datagen = ImageDataGenerator(rescale = 1./255.,
                                     rotation_range = 40,
                                     width_shift_range = 0.2,
                                     height_shift_range = 0.2,
                                     shear_range = 0.2,
                                     zoom_range = 0.2,
                                     horizontal_flip = True,
                                     fill_mode = 'nearest')

  # Pass in the appropriate arguments to the flow_from_directory method
  train_generator = train_datagen.flow_from_directory(directory = TRAINING_DIR,
                                                      batch_size = 45,
                                                      class_mode = 'binary',
                                                      target_size = (150, 150))

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  validation_datagen = ImageDataGenerator(rescale = 1./255.)

  # Pass in the appropriate arguments to the flow_from_directory method
  validation_generator = validation_datagen.flow_from_directory(directory = VALIDATION_DIR,
                                                                batch_size = 5,
                                                                class_mode = 'binary',
                                                                target_size = (150, 150))
  ### END CODE HERE
  return train_generator, validation_generator

# grader-required-cell

# Test your generators
train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)


def create_model():
  # DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
  # USE AT LEAST 3 CONVOLUTION LAYERS

  ### START CODE HERE

  model = tf.keras.models.Sequential([
      # Note the input shape is the desired size of the image 150x150 with 3 bytes color
      tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      # Flatten the results to feed into a DNN
      tf.keras.layers.Flatten(),
      # 512 neuron hidden layer
      tf.keras.layers.Dense(512, activation='relu'),
      # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  from tensorflow.keras.optimizers import RMSprop

  model.compile(optimizer=RMSprop(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

  ### END CODE HERE

  return model

"""Now it is time to train your model!

Note: You can ignore the `UserWarning: Possibly corrupt EXIF data.` warnings.
"""

# Get the untrained model
model = create_model()

history = model.fit(train_generator,
                    epochs=15,
                    verbose=1,
                    validation_data=validation_generator)

model.save('gender_detection2.model')

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.show()
print("")

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.show()