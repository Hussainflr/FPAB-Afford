import os
import numpy as np
import random
import torch
import colorsys
from PIL import Image
from tqdm import trange
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms as tf
from .base import BaseDataset


class COCOSegmentation(BaseDataset):
    NUM_CLASS = 9
    CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def __init__(self,
                 root=os.path.expanduser('/BigDisk/hussain/code/segmentation/FastFCN/encoding/data/updated_FPHA-Afford'),
                 split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(COCOSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        from pycocotools.coco import COCO
        from pycocotools import mask
        if split == 'train':
            print('train set')
            ann_file = os.path.join(root, 'annotations/train.json')
            ids_file = os.path.join(root, 'annotations/train_ids.pth')
            self.root = os.path.join(root, 'train')
        else:
            print('val set')
            ann_file = os.path.join(root, 'annotations/test.json')
            ids_file = os.path.join(root, 'annotations/val_ids.pth')
            self.root = os.path.join(root, 'test')
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.transform = transform
        self.target_transform = target_transform
        self.counter = 0
        self.rootdir = '/BigDisk/hussain/code/segmentation/FastFCN/experiments/segmentation/results/ori'

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        
        #tem = Image.open(os.path.join(self.root,path)).convert('RGB')
        
        img = cv2.imread(os.path.join(self.root, path), 1)
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width'])
        
        #croping image with respect to new dilated bounding box
        



        img , mask = self.crop_image_mask(cocotarget, img, mask)
        #img.save(os.path.join(self.rootdir, str(self.counter)+'.jpeg'))
        #self.counter +=1
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)

        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        elif self.mode == 'test':
            mask = self._mask_transform(mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        image = img

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        
        image.save("atloader.jpeg")
        return img, mask

    def __len__(self):
        return len(self.ids)

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while." + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def crop_image_mask(self, anns, image, mask):

        pt1 = []
        pt2 = []
        #print("annotations: ", len(anns))
        for ann in anns:
            seg = ann["segmentation"][0]
            
            poly = np.asarray(seg).reshape((int(len(seg) / 2), 2))

            pt1.append(np.min(poly, axis=0).astype(dtype=int))
            pt2.append(np.max(poly, axis=0).astype(dtype=int))








        if len(pt1) > 0 and len(pt2) > 0:

            #print("########Mask and Image dimensions : ",image.shape, mask.shape)
            img = image
            h, w, _ = img.shape

            pts1 = np.min(pt1, axis=0)
            pts2 = np.max(pt2, axis=0)
            center = (int((pts1[0] + pts2[0]) / 2), int((pts1[1] + pts2[1]) / 2))
            xscale = 350
            yscale = 300

            xr = int(xscale * 0.5)
            yr = int(yscale * 0.5)

            # cv2.circle(img, center, 20, (0, 0, 255), -1)

            dpts1 = [pts1[0] - xr, pts1[1] - yr]

            dpts2 = [pts2[0] + xscale, pts2[1] + yscale]

            if dpts1[0] < 0:
                dpts2[0] += int(abs(dpts1[0]) / 2) if dpts2[0] < img.shape[1] else img.shape[1]
                dpts2[1] += int(abs(dpts1[0]) / 2) if dpts2[1] < img.shape[0] else img.shape[0]
                if dpts2[0] > img.shape[1]:
                    dpts2[0] = img.shape[1]
                if dpts2[0] > img.shape[0]:
                    dpts2[0] = img.shape[0]

                dpts1[0] = 0

            if dpts1[1] < 0:
                dpts2[1] += int(abs(dpts1[1]) / 2) if dpts2[1] < img.shape[0] else img.shape[0]
                dpts2[0] += int(abs(dpts1[1]) / 2) if dpts2[0] < img.shape[1] else img.shape[1]
                if dpts2[1] > img.shape[0]:
                    dpts2[1] = img.shape[0]

                if dpts2[0] > img.shape[1]:
                    dpts2[0] = img.shape[1]

                dpts1[1] = 0

            if dpts2[0] > img.shape[1]:
                dpts1[0] -= int((int(dpts2[0] - img.shape[1]) / 2) / 2) if dpts1[0] > 0 else 0
                dpts1[1] -= int(((int(dpts2[0] - img.shape[1]) / 2)) / 2) if dpts1[1] > 0 else 0
                if dpts1[0] < 0:
                    dpts1[0] = 0
                if dpts1[1] < 0:
                    dpts1[1] = 0

                dpts2[0] = img.shape[1]
                
                
            if dpts2[1] > img.shape[0]:
                
                dpts1[1] -= int(((dpts2[1] - img.shape[0]) / 2) / 2) if dpts1[1] > 0 else 0
                dpts1[0] -= int(((dpts2[1] - img.shape[0]) / 2) / 2) if dpts1[0] > 0 else 0
                if dpts1[1] < 0:
                    dpts1[1] = 0
                if dpts1[0] < 0:
                    dpts1[0] = 0
                dpts2[1] = img.shape[0]
                # dpts1[1] -= int(/2) if dpts1[1] < 0 else 0

            # print("All:", dpts1, dpts2
            #       )

            #cv2.rectangle(img, tuple(pts1), tuple(pts2), (0, 255, 0), thickness=10)
            #cv2.rectangle(img, tuple(dpts1), tuple(dpts2), (255, 0, 0), thickness=10)
            #cv2.rectangle(mask, tuple(pts1), tuple(pts2), (0, 0, 0), thickness=10)
            #cv2.rectangle(mask, tuple(dpts1), tuple(dpts2), (0, 0, 0), thickness=10)



            # crop image and depth
            cropimg = img[dpts1[1]: dpts2[1], dpts1[0]: dpts2[0]]
            cropmask = mask[dpts1[1]:dpts2[1], dpts1[0]:dpts2[0]]

            #print("dimensions of cropimage and cropmask:",cropimg.shape, cropmask.shape)
            
            #colors = self.random_colors()
            #blendimg = self.apply_mask(cropimg, cropmask, colors)
            #plt.figure("Masked Image")
            #plt.imshow(cropmask)

            #plt.savefig("mask.jpg")
            #plt.imshow(blendimg)
            #plt.savefig("oriimg.jpg")
            #exit()
            #print("***************")
            #cropmask = cv2.cvtColor(cropmask, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(cropimg,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)                  
            mask = Image.fromarray(cropmask)
            img.save("mask.jpeg")
            #print("mask..", type(img), type(mask.size))



            ch, cw, _ = cropimg.shape

            xratio = w / cw
            yratio = h / ch

        return img, mask

    def apply_mask(self, image, mask, color=None, alpha=0.5):
        """Apply the given mask to the image.
        """

        if color is not None:
            for c in range(3):
                image[:, :, c] = np.where(mask == 1,
                                          image[:, :, c] *
                                          (1 - alpha) + alpha * color[c] * 255,
                                          image[:, :, c])
            return image
        else:
            image =  image *(1.0 - alpha) + mask * alpha

        return image

    def random_colors(self, N=11, bright=True):
        
        brightness = 1.0 if bright else 0.7

        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)

        return colors


"""
NUM_CHANNEL = 91
[] background
[5] airplane
[2] bicycle
[16] bird
[9] boat
[44] bottle
[6] bus
[3] car
[17] cat
[62] chair
[21] cow
[67] dining table
[18] dog
[19] horse
[4] motorcycle
[1] person
[64] potted plant
[20] sheep
[63] couch
[7] train
[72] tv
"""
