# -*- coding: utf-8 -*-
import json
import os

import cv2

from extract_features import VGGNet
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

b_annotations = '../data/test/b_annotations.json'
with open(b_annotations, "r") as f:
    anno = json.load(f)


query = '../data/test/cut_images/'
index = './vgg_featureCNN.h5'
result = '../data/test/b_images/'
# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(index, 'r')
# feats = h5f['dataset_1'][:]
feats = h5f['dataset_1'][:]
# print(feats)
imgNames = h5f['dataset_2'][:]
# print(imgNames)
h5f.close()

print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")

# init VGGNet16 model
model = VGGNet()

count = 0

# read and show query image
# queryDir = args["query"]
category_dict = {}

for q in os.listdir(query):

    # extract query image's feature, compute simlarity score and sort
    queryVec = model.vgg_extract_feat(os.path.join(query + q))
    scores = np.dot(queryVec, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    # print (rank_ID)
    # print(rank_score)

    # number of top retrieved images to show
    maxres = 3
    imlist = [str(imgNames[index], encoding="utf-8") for i, index in enumerate(rank_ID[0:maxres])]

    # find the image corresponding  category_id
    for img in anno["images"]:
        if img["file_name"] == imlist[0]:
            category_dict = [(a["id"], a["category_id"]) for a in anno["annotations"]
                             if a["id"] == img["id"]]


    # load boundingbox
    with open("../submit/predictions_boundingbox.json", "r", encoding="utf-8") as m:
        predictions = json.load(m)
        for p in predictions["annotations"]:
            for v, k in category_dict:
                if p["image_id"] == v and p["bbox"] == q.split("_")[1]:
                    predictions["annotations"]["category_id"] = k

                    # write the corresponding category to json file (final result)
                    with open("../submit/predictions_boundingbox.json", "w") as h:
                        json.dump(predictions, m)

    count += 1
    print(count)



        # print(type(imgNames[index]))
    #     print("image names: " + str(imgNames[index]) + " scores: %f" % rank_score[i])
    # print("top %d images in order are: " % maxres, imlist)

    # # show top #maxres retrieved result one by one
    # for i, im in enumerate(imlist):
    #     image = mpimg.imread(result + "/" + str(im, 'utf-8'))
    #     plt.title("search output %d" % (i + 1))
    #     plt.imshow(image)
    #     plt.show()
