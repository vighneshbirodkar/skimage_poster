from skimage import data, io, feature, color, draw
import numpy as np
from matplotlib import pyplot as plt

image = data.hubble_deep_field()[0:500, 0:500]
image_gray = color.rgb2gray(image)

blobs = feature.blob_doh(image_gray, max_sigma=30, threshold=.005)

fig, ax = plt.subplots(1, 1)
ax.imshow(image)
for blob in blobs:
    r, c, radius = blob
    c = plt.Circle((c, r), radius, color='lime', linewidth=2, fill=False)
    ax.add_patch(c)

plt.axis('off')
plt.savefig('blobs.png', bbox_inches='tight', pad_inches=0)
