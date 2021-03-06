import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data 
from PIL import Image as PILImage
 
 
class LIPParsingEdgeDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(473, 473)
                 , mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror 
        
        self.img_ids = []
        tmp_img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if not max_iters==None:
            #self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            for i in range(int(np.ceil(float(max_iters) / len(tmp_img_ids)))):
                np.random.shuffle(tmp_img_ids)
                self.img_ids.extend(tmp_img_ids)
                
        self.files = []
         
        for item in self.img_ids:
            image_path, label_path, label_rev_path, edge_path = item
            name = osp.splitext(osp.basename(label_path))[0]  
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path) 
            label_rev_file = osp.join(self.root, label_rev_path)
            edge_file = osp.join(self.root, edge_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "label_rev": label_rev_file, 
                "edge": edge_file,
                "name": name
            })
          
    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label
 
    def generate_scale_label_pose(self, image, label, edge):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        edge = cv2.resize(edge, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
         
        return image, label, edge
     
    def __getitem__(self, index):
        datafiles = self.files[index]
          
        name = datafiles["name"]  
         
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(datafiles["edge"], cv2.IMREAD_GRAYSCALE)
        edge[edge==255] = 1
        label_rev = cv2.imread(datafiles["label_rev"], cv2.IMREAD_GRAYSCALE)
         
        size = image.shape 
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip, :] 
            edge = edge[:, ::flip] 
            if flip == -1:
                label = label_rev 

        if self.scale:
            image, label, edge = self.generate_scale_label_pose(image, label, edge)
            
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
            edge_pad = cv2.copyMakeBorder(edge, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0,))
        else:
            img_pad, label_pad, edge_pad = image, label, edge

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        edge = np.asarray(edge_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1)) # HWC to CHW
        
        return image.copy(), label.copy(), edge.copy(), np.array(size), name    
  
class LIPDataValSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(473, 473), mean=(128, 128, 128)):
        self.root = root 
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
         
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        self.files = [] 

        for item in self.img_ids:
            image_path, label_path = item
            name = osp.splitext(osp.basename(image_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path) 
            self.files.append({
                "img": img_file,  
                "label": label_file,
                "name": name
            }) 
    def generate_scale_image(self, image, f_scale): 
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR) 
        return image
    
    def resize_image(self, image, size): 
        image = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR) 
        return image
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)   
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        ori_size = image.shape
        image = self.resize_image(image, (self.crop_h, self.crop_w))
         
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        image -= self.mean
         
        image = image.transpose((2, 0, 1))
        return image, label,  np.array(ori_size), name
     
 
class LIPDataTestSet(data.Dataset):
    def __init__(self, root, ori_root, list_path, crop_size=(505, 505), mean=(128, 128, 128)):
        self.root = root
        self.ori_root = ori_root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
         
        self.img_ids = [i_id.strip().split()[0] for i_id in open(list_path)]
        self.files = []  
        for image_path in self.img_ids:
            name = osp.splitext(osp.basename(image_path))[0]
            img_file = osp.join(self.root, image_path)
            ori_img_file = osp.join(self.ori_root, image_path)
            self.files.append({
                "img": img_file,
                "ori_img":ori_img_file
            })

    def __len__(self):
        return len(self.files)
    
    def generate_scale_image(self, image, f_scale): 
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR) 
        return image
    
    def __getitem__(self, index):
        datafiles = self.files[index] 
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) 
        ori_image = cv2.imread( datafiles["ori_img"], cv2.IMREAD_COLOR)
         
        ori_size = ori_image.shape
        size = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean
        
        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
        image = image.transpose((2, 0, 1))
        return image, name, np.array(size), np.array(ori_size)
    
if __name__ == '__main__':
    dst = LIPParsingEdgeDataSet('/home/amax/LIP/LIP_data_model/data/resize321_681/split_321_681', './list/lip/trainList_rev_heat_split.txt')
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels, heatmaps,_,_ = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            cv2.imshow('',img)
            cv2.waitKey()
