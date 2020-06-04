# -*- coding: utf-8 -*-
"""
Multiple funtions created to process the dataset

Created on Wed Jun  3 19:41:27 2020

@author: josea
"""
import glob
import numpy as np
from pathlib import Path
import h5py
from skimage import io
import os
from skimage.transform import resize


def load_SVHN(folder_path):
    """ Load images stored in hdf files, if there isn't hdf files it will try 
        to create new hdf files from uncompresed images
        Parameters:
        ---------------
        folder_path   path where files are located
        
        Returns:
        ----------
        data      Array of images (data[0])
                  Array of labels (data[1])
    """
    print("loading data from "+folder_path)
    nh5 = len(glob.glob1(folder_path,"*.h5"))
    try:
        imgs, lbls  = read_many_hdf5(folder_path, 0)
        for i in range(1,nh5):
            images, labels  = read_many_hdf5(folder_path, i)
            imgs = np.append(imgs,images, axis=0)
            lbls = np.append(lbls,labels, axis=0)
        data = [imgs,lbls]
    except:
        print("Archivo no existe, generando nuevos datos ...")
        read_data(folder_path)
        imgs, lbls  = read_many_hdf5(folder_path, 0)
        for i in range(1,nh5):
            images, labels  = read_many_hdf5(folder_path, i)
            imgs = np.append(imgs,images, axis=0)
            lbls = np.append(lbls,labels, axis=0)
        data = [imgs,lbls]
    return data

def read_many_hdf5(path, name):
    """ Reads image and labels from HDF5.
        Parameters:
        ---------------
        path   path where the hdf file is located
        name   name of the hdf file

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    hdf5_dir = Path(path)
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{name}.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")

    return images, labels

def store_many_hdf5(images, labels, path, name):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    hdf5_dir = Path(path)
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{name}.h5", "w")

    # Create a dataset in the file
    file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()

def read_data(path):
    """ Reads svhn .mat files, crop the images to show just the numbers and 
        store in a new hdf file
        Parameters:
        ---------------
        path   folder path where the .mat file is located
    """
    with h5py.File(path+"/digitStruct.mat", "r") as f:    
        n_data = f["/digitStruct/bbox"].len()
        img_data = []
        lbl_data = []
        n=0
        print("creando archivo de imagenes")
        for i in range(n_data):
            image, label, boxes = get_boxes(path, i, f)
            if(len(boxes)<=5):
                img_data.append(image)
                lbl_data.append(label)
            if(i%10000==9999):
                print("guardando archivo de iamgenes....")
                store_many_hdf5(img_data, lbl_data, path, n)
                img_data = []
                lbl_data = []
                n = n + 1
        store_many_hdf5(img_data, lbl_data, path, n)
    
def get_boxes(path, index, f):
    """ Get the borders of the image, crop it and resize it to 64 x 64
        Parameters:
        ---------------
        path    folder path where the images are
        index   index of the image to process
        f       hdf5 object
        Returns:
        ----------
        image       Cropped image
        labels      Numerical value of the image
        boxes       Borders of the image containing the numbers
    """
    boxes = []
    labels = np.asarray([])
    bbox_data = f["/digitStruct/bbox"]
    n_boxes = f[bbox_data[index][0]]["label"].len()
    image_path = ""
    if(n_boxes == 1):
        box = {}
        box['height'] = f[bbox_data[index][0]]["height"][0][0]
        box['label'] = f[bbox_data[index][0]]["label"][0][0]
        labels = np.append(labels,0) if (box['label'] == 10) else np.append(labels,box['label'])
        box['left'] = f[bbox_data[index][0]]["left"][0][0]
        box['top'] = f[bbox_data[index][0]]["top"][0][0]
        box['width'] = f[bbox_data[index][0]]["width"][0][0]
        box['name'] = str(index+1) + ".png"
        image_path = str(index+1) + ".png"
        boxes.append(box)
    else:
        for i in range(n_boxes):
            box = {}
            box['height'] = f[f[bbox_data[index][0]]["height"][i][0]][()][0][0]
            box['label'] = f[f[bbox_data[index][0]]["label"][i][0]][()][0][0]
            labels = np.append(labels,0) if (box['label'] == 10) else np.append(labels,box['label'])
            box['left'] = f[f[bbox_data[index][0]]["left"][i][0]][()][0][0]
            box['top'] = f[f[bbox_data[index][0]]["top"][i][0]][()][0][0]
            box['width'] = f[f[bbox_data[index][0]]["width"][i][0]][()][0][0]
            box['name'] = str(index+1) + ".png"
            image_path = str(index+1) + ".png"
            boxes.append(box)
    image = io.imread(os.path.join(path,image_path))
    bounds = get_borders(boxes)
    y1 = int(bounds[1]) if int(bounds[1])>=0 else 0
    x1 = int(bounds[3]) if int(bounds[3])>=0 else 0
    y2 = int(bounds[0]) if int(bounds[0])>=0 else 0
    x2 = int(bounds[2]) if int(bounds[2])>=0 else 0
    image = image[y1:x1, y2:x2]
    
    image = np.multiply(resize(image, (64, 64)),255.0).tolist()
    for i in range(5 - len(labels)):
        labels = np.insert(labels, 0, 10, 0)
    return image,labels,boxes

def get_borders(img_data):
    x1 = 9999999
    y1 = 9999999
    x2 = 0
    y2 = 0
    for box in img_data:
        if(box["top"]<y1): y1 = box["top"] 
        if(box["left"]<x1): x1 = box["left"]
        if(box["top"] + box["height"] > y2): y2 = box["top"] + box["height"]
        if(box["left"] + box["width"] > x2): x2 = box["left"] + box["width"]

    return [x1, y1, x2, y2]