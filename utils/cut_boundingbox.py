import cv2
import numpy as np
import os
import json


cut_path = "../data/test/cut_images/"
def main():
    root = "../data/test/"
    with open("../submit/predictions_boundingbox.json") as p:
        predictions = json.load(p)

    images = predictions["images"]
    annotations = predictions["annotations"]

    imagelist = [(pr["file_name"], pr["id"]) for _, pr in enumerate(images)]
    for img, id in imagelist:
        img_path = os.path.join(root + "a_images/" + img)
        im = cv2.imread(img_path)
        objects = [ob["bbox"] for ob in annotations if id == ob["image_id"]]


        for _, object in enumerate(objects):
            xmin, ymin, w, h = int(object[0]), int(object[1]), int(object[2]), int(object[3])
            img_cut = im[ymin: ymin+h, xmin: xmin+w, :]
            if img_cut.size != 0:
                cv2.imwrite(os.path.join(cut_path + "{}_{}.jpg"
                                         .format(id, [xmin, ymin, w, h])), img_cut)
                print(id)


if __name__ == '__main__':
    main()