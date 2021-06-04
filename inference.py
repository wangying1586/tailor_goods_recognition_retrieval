import glob
import os

from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torch

# Label map
voc_labels = 'object'
label_map = {voc_labels: 1}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


# Load model checkpoint
checkpoint = './checkpoint_ssd300_classify_2_1400.pth.tar'
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def run(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0]
    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    #for l in det_labels[0]:
        #print(l)
    det_labels = [rev_label_map[l] for l
                  in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(str(det_labels[i]).upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=str(det_labels[i]).upper(), fill='white',
                  font=font)
    del draw

    return annotated_image, det_boxes, det_labels, det_scores


if __name__ == '__main__':
    number = 0
    root = './data/test/'
    images = []
    annotations = []
    import json
    with open(os.path.join(root + "a_annotations.json"), "r") as f:
        dict = json.load(f)["images"]
        # 遍历文件
        for v, k in enumerate(dict):
            original_image = Image.open(os.path.join(root + "a_images/" + k["file_name"]), mode='r')
            original_image = original_image.convert('RGB')
            visuialization, det_boxes,\
            det_labels, det_scores = run(original_image, min_score=0.01,
                                         max_overlap=0.0, top_k=30)
            # visuialization.show()

            images.append({"file_name": k["file_name"], "id": k["id"]})
            for n in range(len(det_labels)):
                if torch.is_tensor(det_boxes):
                    det_boxes = det_boxes.tolist()
                det_boxes[n][0], det_boxes[n][1],\
                det_boxes[n][2], det_boxes[n][3] = int(det_boxes[n][0]), int(det_boxes[n][1]), \
                                                   int(det_boxes[n][2]-det_boxes[n][0]), \
                                                   int(det_boxes[n][3]-det_boxes[n][1])
                annotations.append({"image_id": k["id"],
                                    "bbox": det_boxes[n],
                                    "category_id": det_labels[n],
                                    "score": round(det_scores[0][n].item(), 2)})
            number += 1
            print(number)
    predictions = {"images": images, "annotations": annotations}
    with open("./submit/predictions_boundingbox.json", "w") as f:
        json.dump(predictions, f)

    # img = {"image": images}
    # with open("images.json", "w") as f:
    #     json.dump(img, f)
    # anno = {"annotations": annotations}
    # with open("predictions.json", "w") as f:
    #     json.dump(anno, f)
