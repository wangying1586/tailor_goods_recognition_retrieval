# coding:utf-8

import torch

from torch import nn
from utils.utils import *
import torch.nn.functional as F
from math import sqrt
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGBase(nn.Module):

    def __init__(self):
        super(VGGBase, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1)

        # Replacements for FC6 and FC7 ini VGG16
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=6, dilation=6)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """

        out = F.relu(self.conv1_1(image))  # (N, 64, 300, 300)  2 64 512 512
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)  2 64 512 512
        out = self.pool1(out)  # (N, 64, 150, 150)  2 64 256 256

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)  2 128 256 256
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)  2 128 256 256
        out = self.pool2(out)  # (N, 128, 75, 75)  2 64 128 128

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)  2 256 128 128
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)  2 256 128 128
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)  2 256 128 128
        conv3_3_feats = out  # 2 256 128 128
        out = self.pool3(out)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True  2 256 64 64

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)  2 512 64 64
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)  2 512 64 64
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)  2 512 64 64
        # conv4_3_feats = out  # (N, 512, 38, 38) 2 512 64 64
        out = self.pool4(out)  # (N, 512, 19, 19)   2 512 32 32

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)   2 512 32 32
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)   2 512 32 32
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)   2 512 32 32
        out = self.pool5(out)  # (N, 512, 19, 19), pool5 does not reduce dimensions   2 512 32 32

        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19) 2 1024 32 32

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 19, 19)  2 1024 32 32

        # Lower-level feature maps
        return conv3_3_feats, conv7_feats

    def load_pretrained_layers(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrainied_params_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):
            state_dict[param] = pretrained_state_dict[pretrainied_params_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']   # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")


class AuxiliaryConvolutions(nn.Module):
    """
    Addititonal convolutios to produce higher-level feature maps
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv12_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.conv12_2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)  256 32 32
        out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)   512 16 16
        conv8_2_feats = out  # (N, 512, 10, 10)  512 16 16

        out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)  128 16 16
        out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)  256 8 8
        conv9_2_feats = out  # (N, 256, 5, 5)  256 8 8

        out = F.relu(self.conv10_1(out))  # (N, 128, 5, 5)  128 8 8
        out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)  256 4 4
        conv10_2_feats = out  # (N, 256, 3, 3)  256 4 4

        out = F.relu(self.conv11_1(out))  # (N, 128, 3, 3)  128 4 4
        conv11_2_feats = F.relu(self.conv11_2(out))  # (N, 256, 1, 1)  256 2 2

        out = F.relu(self.conv12_1(conv11_2_feats))  # 128 2 2
        conv12_2_feats = F.relu(self.conv12_2(out))  # n 256 1 1

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats, conv12_2_feats


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predictt class scores and bounding boxes
    using lower and higher-level feature maps

    """

    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv3_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 6,
                   'conv11_2': 4,
                   'conv12_2': 4}

        # Localization prediction convolutions
        self.loc_conv3_3 = nn.Conv2d(256, n_boxes['conv3_3'] * 4, kernel_size=3, stride=1, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3,  stride=1, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, stride=1,  padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, stride=1,  padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, stride=1,  padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, stride=1,  padding=1)
        self.loc_conv12_2 = nn.Conv2d(256, n_boxes['conv12_2'] * 4, kernel_size=3, stride=1,  padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv3_3 = nn.Conv2d(256, n_boxes['conv3_3'] * n_classes, kernel_size=3, stride=1,  padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, stride=1,  padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, stride=1,  padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, stride=1,  padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, stride=1,  padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, stride=1,  padding=1)
        self.cl_conv12_2 = nn.Conv2d(256, n_boxes['conv12_2'] * n_classes, kernel_size=3, stride=1,  padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)


    def forward(self, conv3_3_feats, conv7_feats, conv8_2_feats,
                conv9_2_feats, conv10_2_feats, conv11_2_feats, conv12_2_feats):

        batch_size = conv3_3_feats.size(0)

        l_conv3_3 = self.loc_conv3_3(conv3_3_feats)  # (N, 16, 38, 38)
        l_conv3_3 = l_conv3_3.permute(0, 2, 3, 1).contiguous()

        l_conv3_3 = l_conv3_3.view(batch_size, -1, 4)  # (N, 5776, 4)

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

        l_conv12_2 = self.loc_conv12_2(conv12_2_feats)  # (N, 16, 1, 1)
        l_conv12_2 = l_conv12_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv12_2 = l_conv12_2.view(batch_size, -1, 4)  # (N, 4, 4)

        # Predict classes in localization boxes
        c_conv3_3 = self.cl_conv3_3(conv3_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv3_3 = c_conv3_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv3_3 = c_conv3_3.view(batch_size, -1,
                                   self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        c_conv12_2 = self.cl_conv12_2(conv12_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv12_2 = c_conv12_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv12_2 = c_conv12_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        # A total of 8732 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv3_3, l_conv7,
                          l_conv8_2, l_conv9_2,
                          l_conv10_2, l_conv11_2, l_conv12_2], dim=1)  # (N, 8732, 4)
        classes_socres = torch.cat([c_conv3_3, c_conv7,
                                    c_conv8_2, c_conv9_2,
                                    c_conv10_2, c_conv11_2, c_conv12_2], dim=1)  # (N, 8732, n_classes)

        return locs, classes_socres


class SSD300(nn.Module):

    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 256, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        # 2 256 125 125    2 1024 31 31
        conv3_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38)

        # Rescale conv4_3 after L2 norm
        norm = conv3_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv3_3_feats = conv3_3_feats / norm  # (N, 512, 38, 38)   2 256 125 125
        conv3_3_feats = conv3_3_feats * self.rescale_factors  # (N, 512, 38, 38)  2 256 125 125
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats, conv12_2_feats = \
            self.aux_convs(conv7_feats)
        # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions
        # (predict offsets w.r.t prior-boxes in each resulting localization box)
        locs, classes_scores = self.pred_convs(conv3_3_feats,
                            conv7_feats,
                            conv8_2_feats,
                            conv9_2_feats,
                            conv10_2_feats,
                            conv11_2_feats,
                            conv12_2_feats)
        # (N, 8732, 4)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper

        :return: prior boxes in center-size coordinates, a tensor of dimensions
        (8732, 4)
         """
        fmap_dims = {'conv3_3': 64,
                     'conv7': 32,
                     'conv8_2': 16,
                     'conv9_2': 8,
                     'conv10_2': 4,
                     'conv11_2': 2,
                     'conv12_2': 1}

        obj_scales = {'conv3_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.325,
                      'conv9_2': 0.5,
                      'conv10_2': 0.675,
                      'conv11_2': 0.825,
                      'conv12_2': 0.9}

        aspect_ratios = {'conv3_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5],
                         'conv12_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap]
                        * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        """
                        对于长宽比为1的情况，使用一个额外的先验，其尺度为
                        当前特征图的比例和下一个特征图的比例的几何平均值
                        """
                        if ratio == 1:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])

                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0, 1)   # compact tensor to [0,1]

        return prior_boxes


    def detect_objects(self, predicted_locs, preditcted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        preditcted_scores = F.softmax(preditcted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == preditcted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_labels = preditcted_scores[i].max(dim=1)  # 8732

            # Check for each class
            for c in range(0, self.n_classes):
                # Keep only predicted boxes and scores
                # where scores for this class are above the minimum scores
                class_scores = preditcted_scores[i][:, c]
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # n_qualified n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # n_qualified, 4

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # n_qualified, n_min_score
                class_decoded_locs = class_decoded_locs[sort_ind]  # n_min_score, 4

                # Find the overlap between predicted boxes
                # n_qualified n_min_score
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)

                # NMS

                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes
                    # like an "or" operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
        The MultiBox loss, a loss function for object detection.

        This is a combination of:
        (1) a localization loss for the predicted locations of the boxes, and
        ------(2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.prior_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :return: multibox loss
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)

        # For each image
        for i in range(batch_size):

            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i], self.prior_xy)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            """
            # 我们不希望出现一个物体在我们的正向（非背景）先验中没有被代表的情况 --
            # 1. 一个对象可能不是所有优先权的最佳对象，因此不在 object_for_each_prior 中。
            # 2. 所有带有该对象的先验指标可能被分配为基于阈值（0.5）的背景。

            # 为了补救这个问题 --
            # 首先，找到对每个物体有最大重叠的先验。
            """
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # 8732
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # 8732

            # Store
            true_classes[i] = label_for_each_prior

            # (8732, 4)
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[i], true_locs[i])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        print(predicted_scores.shape, true_classes.shape)
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes),
                                           true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss


if __name__ == '__main__':
    img = torch.ones((2, 3, 512, 512))
    model = SSD300(n_classes=2)
    locs, classname = model(img)
    print(locs, classname)





