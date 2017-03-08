import io
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import pywt
import numpy as np
import pywt
import cv2
from scipy import signal

def get_image():
    image = Image.open(image_name)
    img_grey = image.convert('L')
    img = np.array(img_grey, dtype=np.float)
    return img

def get_2D_dct(img):
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')

def get_2d_idct(coefficients):
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')


def get_reconstructed_image(raw):
    img = raw.clip(0, 255)
    img = img.astype('uint8')
    img = Image.fromarray(img)
    return img

def dct(image_name):
    import os
    original_size =  os.stat(image_name).st_size
    pixels = get_image()
    dct_size = pixels.shape[0]
    dct = get_2D_dct(pixels)
    reconstructed_images = []
    compression_ratio = []
    compression_error = []
    for ii in range(dct_size):
        dct_copy = dct.copy()
        dct_copy[ii:,:] = 0
        dct_copy[:,ii:] = 0
        r_img = get_2d_idct(dct_copy)

        reconstructed_image = get_reconstructed_image(r_img)
        f = open('data/dct_kitten_cv{}.jpg'.format(ii),'wb')
        reconstructed_image.save(f)
        image_size = os.stat('data/kitten_cv{}.jpg'.format(ii)).st_size
        ratio = float(original_size)/float(image_size)
        compression_ratio.append(ratio)
        err = np.sum((pixels - reconstructed_image) ** 2)
        err = err/float(pixels.shape[0] * pixels.shape[1])
        compression_error.append(err)
        reconstructed_images.append(reconstructed_image)


    # fig = plt.figure(figsize=(16, 16))
    # for ii in range(16):
    #     plt.subplot(4, 4, ii + 1)
    #     plt.imshow(reconstructed_images[ii*4], cmap=plt.cm.gray)
    #     plt.grid(False)
    #     plt.xticks([])
    #     plt.yticks([])

    # plt.plot([i for i in range(0,len(compression_ratio))],compression_ratio)
    # plt.plot([i for i in range(0,len(compression_error))],compression_error)
    plt.scatter(compression_ratio,compression_error)
    plt.show()

# image_name = "tree.jpg"
# dct(image_name)
