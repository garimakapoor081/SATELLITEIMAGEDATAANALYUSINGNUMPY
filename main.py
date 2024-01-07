import numpy as np
import matplotlib.pyplot as plt
import scipy
import imageio
import skimage
import warnings
warnings.filterwarnings("ignore")
from skimage import data
photo_data =imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
type(photo_data)
plt.figure(figsize=(15,15))
plt.imshow(photo_data)
plt.show()
print(photo_data)
print(photo_data.shape)
photo_data.min(),photo_data.max()
photo_data.mean()
photo_data[150, 150]=1
photo_data[150,250,1]=1
photo_data[1,1]=1

#Set a Pixel to All Zeros
photo_data = imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
photo_data[150, 250] = 0 # We set all three layers of RGB of this
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
plt.show()
#exclusive
photo_data = imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
photo_data[150:800, :,1] = 500# We set all three layers of RGB of this
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
plt.show()

photo_data = imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
photo_data[150:800, : 1] = 500# We set all three layers of RGB of this
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
plt.show()

photo_data = imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
photo_data[150:800, :] = 500# We set all three layers of RGB of this
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
plt.show()

photo_data = imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
photo_data[150:800,:] = 0# We set all three layers of RGB of this
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
plt.show()
#Pick all Pixels with Low Values

photo_data = imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
print("Shape of photo_data:", photo_data.shape)
low_value_filter = photo_data < 100
print("Shape of low_value_filter:", low_value_filter.shape)
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
photo_data[low_value_filter] = 0 #set low values to 0
plt.figure(figsize=(10,10))
plt.imshow(photo_data)
plt.show()

rows_range = np.arange(len(photo_data)) #Create a range array
print(rows_range)
cols_range = rows_range #Create a range array
print(cols_range)
print(type(rows_range))

#We are setting the selected rows and columns to the maximum value of
photo_data[rows_range, cols_range] = 500
print(photo_data)
plt.figure(figsize=(15,15))
plt.imshow(photo_data)
plt.show()

#Masking Images
#Now let us try to mask the image in shape of a circular disc.
total_rows, total_cols, total_layers = photo_data.shape
print("photo_data = ", photo_data.shape)
X, Y = np.ogrid[:total_rows, :total_cols]
print("X = ", X.shape, " and Y = ", Y.shape)

from IPython.display import Image
Image("Images/figure.png")
center_row, center_col = total_rows / 2, total_cols / 2
print("center_row = ", center_row, "AND center_col = ", center_col)
#print(X - center_row)
#print(Y - center_col)
dist_from_center = (X - center_row)**2 + (Y - center_col)**2
#print(dist_from_center)
radius = (total_rows / 2)**2
#print("Radius = ", radius)
circular_mask = (dist_from_center > radius)
#print(circular_mask)
print(circular_mask[1500:1700,2000:2200])
photo_data = imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
photo_data[circular_mask] = 0
plt.figure(figsize=(15,15))
plt.imshow(photo_data)
#Further Masking

X, Y = np.ogrid[:total_rows, :total_cols]
half_upper = X < center_row # this line generates a mask for all rows

half_upper_mask = np.logical_and(half_upper, circular_mask)
photo_data = imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
photo_data[half_upper_mask] = 500
#photo_data[half_upper_mask] = random.randint(200,255)
plt.figure(figsize=(15,15))
plt.imshow(photo_data)
plt.show()
photo_data = imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
red_mask = photo_data[:, : ,0] < 150
photo_data[red_mask] = 0
plt.figure(figsize=(15,15))
plt.imshow(photo_data)
plt.show()
#Detecting Highl-GREEN Pixels
photo_data = imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
green_mask = photo_data[:, : ,1] < 150
photo_data[green_mask] = 0
plt.figure(figsize=(15,15))
plt.imshow(photo_data)
plt.show()
#Detecting Highly-BLUE Pixels
photo_data = imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
blue_mask = photo_data[:, : ,2] < 150
photo_data[blue_mask] = 0
plt.figure(figsize=(15,15))
plt.imshow(photo_data)
plt.show()
#Composite mask that takes thresholds on all three layers: RED, GREEN,BLUE
photo_data = imageio.imread('D:\\garima project assimnt\\8 Advance Project Dataset-20231015T165511Z-001 (1)\\sd-3layers.jpg')
red_mask = photo_data[:, : ,0] < 150
green_mask = photo_data[:, : ,1] > 100
blue_mask = photo_data[:, : ,2] < 100
final_mask = np.logical_and(red_mask, green_mask, blue_mask)
photo_data[final_mask] = 0
plt.figure(figsize=(15,15))
plt.imshow(photo_data)
plt.show()