import numpy as np
from PIL import Image
import sys

img = Image.open('static/rescources/1.jpg');
np_array = np.array(img)

np.set_printoptions(threshold=sys.maxsize)
print(np_array[0, 3, :])
# np.savetxt('trial.txt', np_array)