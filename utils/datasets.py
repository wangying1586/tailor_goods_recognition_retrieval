import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import transform

class TailorMachineDataset(Dataset):

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.lower()

        assert self.split in {'train', 'test'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '/a_annotations.json'), 'r') as j:
            dict = json.load(j)
        self.images = dict['images']
        self.objects = dict['annotations']

    def __getitem__(self, i):
        # Read image
        if i == self.images[i]['id']:
            file_name = self.images[i]['file_name']

        image = Image.open(os.path.join('../data/' + self.split +'/a_images/' + file_name), mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels)
        boxes = [a["bbox"] for a in self.objects if a["image_id"] == self.images[i]["id"]]  # (n_objects, 4)
        for _, b in enumerate(boxes):
            boxes[_][2], boxes[_][3] = b[0] + b[2], b[1] + b[3]
        labels = [1 for a in self.objects if a["image_id"] == self.images[i]["id"]]   # (n_objects)  1

        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        # Apply transformations
        image, boxes, labels = transform(image, boxes, labels, split=self.split)

        return image, boxes, labels

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects
        we need a collate function (to be passed to the Dataloader)
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, area
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append((b[2]))

        images = torch.stack(images, dim=0)

        return images, boxes, labels

