import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
import skimage
from z5_dct import get_reconstructed_image


def plt_cwt(image_name, output, wavename):
    imArray = cv2.imread(image_name)
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    imArray =  np.float32(imArray)
    imArray /= 255
    # imArray = skimage.util.random_noise(imArray,mode='s&p',amount =0.2)
    wavename = 'coif1'
    cA,cR = pywt.dwt2(imArray,wavename)
    cAA,cAR = pywt.dwt2(cA,wavename) # cAA,cAH,cAV,cAD
    cAAA,cAAR = pywt.dwt2(cAA,wavename) # cAA,cAH,cAV,cAD
    cAAAA,cAAAR = pywt.dwt2(cAAA,wavename) # cAA,cAH,cAV,cAD
    cAAAAA,cAAAAR = pywt.dwt2(cAAAA,wavename) # cAA,cAH,cAV,cAD
    # Level2=[(cAA,cA[0]), (cA[1],cA[2])]
    # cv2.imshow([Level2,cR[0],(cR[1],cR[2])],'Colormap')
    fig = plt.figure(figsize=(64, 64))
    plot_pic(cAAAAA,1)
    plot_pic(cAAAAR[0],2)
    plot_pic(cAAAAR[1],7)
    plot_pic(cAAAAR[2],8)
    plot_pic(cAAAR[0],9)
    plot_pic(cAAAR[1],14)
    plot_pic(cAAAR[2],15)
    plot_pic(cAAR[0],16)
    plot_pic(cAAR[1],21)
    plot_pic(cAAR[2],22)
    plot_pic(cAR[0],23)
    plot_pic(cAR[1],28)
    plot_pic(cAR[2],29)
    plot_pic(cR[0],30)
    plot_pic(cR[1],35)
    plot_pic(cR[2],36)
    plt.show()
    import os
    original_size =  os.stat(image_name).st_size
    fig = plt.figure(figsize=(64, 64))
    images = []
    images.append(pywt.idwt2((cAAAAA, cAAAAR), wavename, 'smooth'))
    images.append(pywt.idwt2((cAAAA, cAAAR), wavename, 'smooth'))
    images.append(pywt.idwt2((cAAA, cAAR), wavename, 'smooth'))
    images.append(pywt.idwt2((cAA, cAR), wavename, 'smooth'))
    images.append(pywt.idwt2((cA, cR), wavename, 'smooth'))

    compression_ratio = []
    compression_error = []
    for i in range(0,len(images)):
        # plot_pic(images[i],i+1)
        f = open('data/cwt_kitten_{}.jpg'.format(i),'wb')
        get_reconstructed_image(images[i]).save(f)
        data =  get_reconstructed_image(images[i])
        data.save(f)
        imArray_H = cv2.imread('data/cwt_kitten_{}.jpg'.format(i))
        imArray_H = cv2.cvtColor( imArray_H,cv2.COLOR_RGB2GRAY )
        imArray_H =  np.float32(imArray_H)
        imArray_H /= 255
        ratio = float(original_size)/float(os.stat('data/kitten_cv{}.jpg'.format(i)).st_size)
        compression_ratio.append(ratio)
        new_image = cv2.resize(images[i],imArray.shape)
        new_image = np.transpose(new_image)
        err = np.sum((imArray -new_image ) ** 2)
        err = err/float(imArray.shape[0] * imArray.shape[1])
        compression_error.append(err)
    # plt.plot([i for i in range(0,len(compression_ratio))],compression_ratio)
    # plt.plot([i for i in range(0,len(compression_error))],compression_error)
    plt.scatter(compression_ratio,compression_error)
    plt.show()



    # fig = plt.figure(figsize=(64, 64))
    # plot_pic(cAAAA,1)
    # plot_pic(cAAAR[0],2)
    # plot_pic(cAAAR[1],17)
    # plot_pic(cAAAR[2],18)
    # plot_pic(cAAR[0],(3,20))
    # plot_pic(cAAR[1],(33,50))
    # plot_pic(cAAR[2],(35,52))
    # plot_pic(cAR[0],(5,56))
    # plot_pic(cAR[1],(65,116))
    # plot_pic(cAR[2],(69,120))
    # plot_pic(cR[0],(9,128))
    # plot_pic(cR[1],(129,248))
    # plot_pic(cR[2],(137,256))
    # plt.show()
    plt.savefig(output)

def plot_pic(pic,id):
    plt.subplot(6,6, id )
    plt.imshow(pic, cmap=plt.cm.gray)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])



def cwt_ratio(image_name,wavename):
    import os
    original_size =  os.stat(image_name).st_size
    imArray = cv2.imread(image_name)
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    imArray =  np.float32(imArray)
    imArray /= 255
    pixels = imArray
    dct_size = pixels.shape[0]
    # imArray = skimage.util.random_noise(imArray,mode='pepper')
    wavename = 'db1'
    cA = pywt.dwt2(imArray,wavename)

    dct = cA[0]
    reconstructed_images = []
    compression_ratio = []
    compression_error = []
    for ii in range(dct_size):
        dct_copy = dct.copy()
        dct_copy[ii:,:] = 0
        dct_copy[:,ii:] = 0
        # Reconstructed image
        r_img = pywt.waverec2(cA,wavename)
        r_img = r_img
        reconstructed_image = get_reconstructed_image(r_img)
        f = open('data/cwt_kitten_{}.jpg'.format(ii),'wb')
        reconstructed_image.save(f)
        ratio = float(original_size)/float(os.stat('data/kitten_cv{}.jpg'.format(ii)).st_size)
        compression_ratio.append(ratio)
        err = np.sum((pixels - reconstructed_image) ** 2)
        err = err/float(pixels.shape[0] * pixels.shape[1])
        compression_error.append(err)
        # print ratio, err
        # print 8*reconstructed_image.size[0]*reconstructed_image.size[1]
        # Create a list of images
        reconstructed_images.append(reconstructed_image)
plt_cwt('kitten.jpg', 'kitten_inverse_db_1.jpg','db1')
# cwt_ratio('kitten.jpg','db1')