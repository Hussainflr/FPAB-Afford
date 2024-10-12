
import os
import pickle

import torch
import trimesh
import numpy as np
from PIL import Image
import copy
import cv2

from torchvision import transforms
from torchvision import utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torchvision
import json




class UnifiedPoseDataset(Dataset):

    def __init__(self, mode='train', root='../data/First_Person_Action_Benchmark', loadit=False, name=None):

        self.loadit = loadit
        self.temp = 0

        if name is None:
            self.name = mode
        else:
            self.name = name

        self.root = root

        if mode == 'clean':
            self.subjects = [1]

        elif mode == 'test2':
            self.subjects = [2, 5, 4]
        else:
            raise Exception("Incorrect vallue for for 'mode': {}".format(mode))

        subject = "Subject_1"
        subject = os.path.join(root, 'Object_6D_pose_annotation_v1', subject)
        self.actions = os.listdir(subject)
        print(self.actions)
        self.object_names = ['juice', 'liquid_soap', 'milk', 'salt']

        action_to_object = {
            'open_milk': 'milk',
            'close_milk': 'milk',
            'pour_milk': 'milk',
            'open_juice_bottle': 'juice',
            'close_juice_bottle': 'juice',
            'pour_juice_bottle': 'juice',
            'open_liquid_soap': 'liquid_soap',
            'close_liquid_soap': 'liquid_soap',
            'pour_liquid_soap': 'liquid_soap',
            'put_salt': 'salt'
        }

        categories_ids = {
            'grasp':0,
            'wrap-grasp':1,
            'contain':2,
            'openable':3,
        }
        new_image_dir = "E:\Research\Important\Dataset\FPHA-Afford\dataset"


        if not loadit:

            self.temp = 0
            self.samples = {}
            idx = 0
            image_count = 5
            #folder count
            fcount = 0

            #count coco annotation files
            anno_count = 0

            #count annotated images
            anno_image_count = 0
            anno_dic = {}


            for subject in self.subjects:

                subject = "Subject_" + str(subject)
                flist = os.listdir(os.path.join(root, 'Video_files', subject))

                for fname in flist:
                    video_sequences = os.listdir(os.path.join(root,'Video_files', subject, fname))
                    for vs in video_sequences:
                        frames = os.listdir(os.path.join(root,'Video_files', subject, fname, vs, 'color'))
                        c = 0

                        #if you want iterate through all iamges then remove the condition "if frames.__contains__("via_export_coco.json"):"
                        if frames.__contains__("via_export_coco.json"):
                            anno_path = os.path.join(root, 'Video_files', subject, fname, vs, 'color','via_export_coco.json')
                            with open(anno_path, 'r') as f:
                                data = dict(json.load(f))
                                f.close()

                            # let's build single coco annotation file





                            anno_count += 1
                            idx = 0
                            annlist = []
                            imagelist =[]
                            catlist = []

                            #get categories folderwise
                            catdic = {}
                            for cat in data['categories']:
                                catdic.__setitem__(cat['id'], cat['name'])
                                cat['id'] = categories_ids.get(cat['name'])
                                catlist.append(cat)

                            for frame in frames:

                                # file could be an image or an annotation file. Both are specified with extension
                                image_path = os.path.join(root, 'Video_files', subject, fname, vs, 'color', frame)


                                if frame.__contains__('.jpeg'):

                                    # counting total images

                                    # this counter is for annotated images
                                    # c += 1

                                    # img = cv2.imread(image_path, 1)

                                    # subject number
                                    sname = str(subject).replace('Subject_', '')
                                    # frame name
                                    fn = str(frame).replace('.jpeg', '').replace('color_', '')

                                    idx = int(fn)



                                    # # updated frame name
                                    # new_frame_name = sname + str(fcount) + str(vs) + fn
                                    #
                                    # imgdir = os.path.join(new_image_dir, new_frame_name)
                                    # # cv2.imwrite(imgdir, img)

                                    for ann in data['annotations']:

                                        # changing image_id into to Int because VIA annotator gives it in string but in coco format it must be in Int
                                        image_id = int(ann['image_id'])

                                        if idx == image_id:
                                            # updated frame name
                                            new_frame_name = sname + str(fcount) + str(vs) + fn


                                            ann['image_id'] = int(new_frame_name)
                                            ann['segmentation'] = [ann['segmentation']]

                                            ann['id'] = image_count

                                            c = ann['category_id']

                                            #get cat name
                                            catname = catdic.get(c)

                                            #get cat id from defined dictionary "categories_ids"
                                            cat_id = categories_ids[catname]

                                            #update cat id in ann['category_id']
                                            ann['category_id'] = cat_id












                                            annlist.append(ann)



                                            #read frame
                                            img = cv2.imread(image_path, 1)

                                            imgdir = os.path.join(new_image_dir, str(new_frame_name)+'.jpeg')


                                            #save frame in new separat dir
                                            cv2.imwrite(imgdir, img)

                                            #frame counter
                                            image_count += 1
                                    for element in data['images']:

                                        # changing image_id into to Int because VIA annotator gives it in string but in coco format it must be in Int
                                        id = int(element['id'])

                                        if idx == id:
                                            # updated frame name


                                            element['id'] = int(new_frame_name)
                                            element['file_name'] = str(new_frame_name)+'.jpeg'
                                            imagelist.append(element)



                            if anno_dic.__len__() == 0:
                                anno_dic = data
                                anno_dic['annotations'] = annlist
                                anno_dic['images'] = imagelist
                                anno_dic['categories'] = catlist
                                with open(os.path.join(new_image_dir, 'tem.json'), 'w+') as f:
                                    print("***********************************")
                                    json.dump(anno_dic, f)
                                print("%%%%%%%%%%%%%%%%%%%%%%")

                            else:
                                annotations = anno_dic['annotations']
                                images =  anno_dic['images']
                                for ann in annlist:
                                    annotations.append(ann)
                                    print("$$$$$$$$")
                                anno_dic['annotations'] = annotations


                                for element in images:
                                    images.append(element)
                                    print("#######")
                                anno_dic['images'] = images

                                #update categories in annotation file

                                categories = anno_dic['categories']
                                for cat in catlist:
                                    categories.append(cat)


                with open(os.path.join(new_image_dir,'coco.json'), 'w+') as f:
                    print("***********************************")
                    json.dump(anno_dic, f)







                                    # anno_count += 1
                                    #anno_image_count += c


                                # elif frame == 'via_export_coco.json':
                                #     with open(file, 'r') as f:
                                #         data = dict(json.load(f))
                                #     # let's build single coco annotation file
                                #     if anno_dic.__len__() == 0:
                                #         anno_dic = data
                                #         print(anno_dic.keys())
                                #
                                #     # anno_count += 1
                                #     anno_image_count += c


                    fcount += 1

                print("Image count: {} folder count: {} Annotation files count: {}".format(image_count, fcount, anno_count))
                # print("Coco Annotation files: {} images count: {}".format( anno_count, anno_image_count))
                break
                for action in self.actions:
                    pose_sequences = set(
                        os.listdir(os.path.join(root, 'Object_6D_pose_annotation_v1', subject, action)))
                    video_sequences = set(os.listdir(os.path.join(root, 'Video_files', subject, action)))
                    sequences = list(pose_sequences.intersection(video_sequences))
                    for sequence in sequences:
                        frames = len(os.listdir(os.path.join(root, 'Video_files', subject, action, sequence, 'color')))

                        for frame in range(frames):
                            sample = {
                                'subject': subject,
                                'action_name': action,
                                'seq_idx': str(sequence),
                                'frame_idx': frame,
                                'object': action_to_object[action]
                            }
                            # print('Subject: ', subject)
                            # print('action_name: ', action)
                            # print('seq_idx: ', str(sequence))
                            # print('frame_idx: ', frame)
                            # print('object :', action_to_object)
                            print(sample)
                            self.samples[idx] = sample
                            idx += 1

            # print("Number of frames :", idx)



            self.save_samples(mode)


        else:

            self.samples = self.load_samples(mode)

    def load_samples(self, mode):

        with open('../cfg/{}.pkl'.format(mode), 'rb') as f:
            samples = pickle.load(f)
            return samples

    def save_samples(self, mode):

        with open('../cfg/{}.pkl'.format(self.name), 'wb') as f:
            # print("number of samples :" , range(self.samples))
            pickle.dump(self.samples, f)
        print("done")
        exit(1)