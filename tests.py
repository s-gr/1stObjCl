import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, image
import os, random


test_image = image.load_img('./dogscats/test1/' + random.choice(os.listdir('./dogscats/test1/')))
plt.figure()
plt.imshow(test_image)
plt.show()

print(random.choice(os.listdir('./dogscats/test1/')))
