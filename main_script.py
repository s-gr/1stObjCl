from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator, image

# from IPython.display import display
# from PIL import Image

import numpy as np
import os, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Building one CNN with Max Pooling and Flattening to give to common FC Layer
model = Sequential()

model.add(Conv2D(
    32, (3, 3),
    input_shape=(150, 150, 3)
    )
)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Adding the FC Layers
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set Batch Size
batch_size = 16

# rescale images
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_data_gen = ImageDataGenerator(rescale=1./255)

train_set = train_data_gen.flow_from_directory(
    'dogscats/train',
    target_size=(150,150),
    batch_size= batch_size,
    class_mode='binary'
)
test_set = test_data_gen.flow_from_directory(
    'dogscats/valid',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
)

if os.path.isfile('2nd_try.h5'):
    model.load_weights('2nd_try.h5')

history = model.fit_generator(
    train_set,
    steps_per_epoch=2000 // batch_size,
    epochs=50,
    validation_data=test_set,
    validation_steps= 800 // batch_size
)

model.save_weights('2nd_try.h5')

image_path = './dogscats/test1/' + random.choice(os.listdir('./dogscats/test1/'))
plt.figure()
plt.imshow(mpimg.imread(image_path))
plt.show()

test_image = image.load_img(image_path, target_size=(150, 150))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

print(train_set.class_indices)
print(result.shape)

if result >= 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
