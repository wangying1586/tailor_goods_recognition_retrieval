import cv2
import numpy as np
from os import walk
from os.path import join

def create_descriptors(folder):
    files = []
    for (dirpath, dirnames, filenames) in walk(folder):
        files.extend(filenames)
    for f in files:
        if '.jpg' in f:
            save_descriptor(folder, f, cv2.xfeatures2d.SIFT_create())

def save_descriptor(folder, image_path, feature_detector):
    # 判断图片是否为npy格式
    if image_path.endswith("npy"):
        return
    # 读取图片并检查特征
    img = cv2.imread(join(folder, image_path), 0)
    keypoints, descriptors = feature_detector.detectAndCompute(img, None)
    # 设置文件名并将特征数据保存到npy文件
    descriptor_file = image_path.replace("jpg", "npy")
    np.save(join(folder, descriptor_file), descriptors)

if __name__=='__main__':
    path = '../data/train/b_images'
    create_descriptors(path)

