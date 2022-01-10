
"""
Least Sigificant Bit hides images (Replace the three least significant bits of the public image with the three most significant bits of the private image)
"""




import os
import cv2
import sys
from PIL import Image

def getFileList(dir, Filelist, ext=None):
    """
    Gets a list of files in a folder and its subfolders
    input dir：folder root
    input ext: filename extension
    return： filepath list
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist

#private images
imglist=[]
#org_img_folder = '../dataset/cifar-10-batches-py/train/cat/'
for i in range(3,8):
   org_img_folder = '../dataset/MNIST/deal_M/train/'+str(i)+'/'
   img_list = getFileList(org_img_folder, [], 'jpg')
   imglist.extend(img_list)
print('This execution has retrieved ' + str(len(imglist)) + ' images\n')

#public images
list_1=['horse','dog','bird','cat','frog','deer']
imglist2=[]
for i in list_1:
   hide_img_folder = '../dataset/cifar-10-batches-py/train/'+i+'/'
   #imglist2 = getFileList(hide_img_folder, [], 'jpg') #public dataset
   img_list=getFileList(hide_img_folder, [], 'jpg')
   imglist2.extend(img_list)
print('This execution has retrieved ' + str(len(imglist2)) + ' images\n')

j=0
for i in imglist:#private dataset
    imageSource = Image.open(imglist2[j]).convert('RGB')#public dataset
    imageToHide = Image.open(i).convert('RGB')
    width, height = imageSource.size
    width2, height2 = imageToHide.size
    if width != width2 or height != height2:
         sys.exit('Error, images must have same size (width x height)')
    imageResult = Image.new('RGB', (width, height))
    for x in range(width):
        for y in range(height):
            pixelSource = imageSource.getpixel((x, y))  # getpixel:obtain the RGB value of a pixel at a point in the image
            pixelToHide = imageToHide.getpixel((x, y))
            pixelResult = ()
            for a, b in zip(pixelSource, pixelToHide):
                pixelResult += (a & 248 | b >> 5),
            imageResult.putpixel((x, y), pixelResult) #Overrides the value at a pixel position
    # imageResult.save('m'+i.split('\\')[4])
    #imageResult.save(imglist2[j].split('\\')[6])
    #imageResult.save("../dataset/Cover_Img/"+imglist2[j].split('/')[5])
    img_path = "../dataset/LSB_MNIST_37/" +imglist[j].split('/')[5]
    isExists = os.path.exists(img_path)
    if not isExists:
        os.makedirs(img_path)
    imageResult.save("../dataset/LSB_MNIST_37/" +imglist[j].split('/')[5]+'/'+ imglist2[j].split('/')[5])
    j=j+1