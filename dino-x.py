# dds cloudapi for DINO-X
# using supervision for visualization
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk.tasks.v2_task import V2Task
from tqdm import tqdm

from rle_util import rle_to_array
from api_key import API_TOKEN_DDS
"""
Hyper Parameters
"""

# TEXT_PROMPT = "<prompt_free>"  # Prompt-Free does not need text prompt
OUTPUT_DIR = Path("./outputs/prompt_free_detection_segmentation/")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

"""
Prompting DINO-X with Text for Box and Mask Generation with Cloud API
"""

# Step 1: initialize the config
token = API_TOKEN_DDS
config = Config(token)

# Step 2: initialize the client
client = Client(config)


# Step 3: Run V2 task
# if you are processing local image file, upload them to DDS server to get the image url


def infer(img_path=None, save_name=None):
    image_url = client.upload_file(img_path)

    v2_task = V2Task(
        api_path="/v2/task/dinox/detection",
        api_body={
            "model": "DINO-X-1.0",
            "image": image_url,
            "prompt": {
                "type": "universal",
                # "text": TEXT_PROMPT
            },
            "targets": ["bbox", "mask"],
            "bbox_threshold": 0.25,
            "iou_threshold": 0.8
        }
    )

    client.run_task(v2_task)
    result = v2_task.result

    objects = result["objects"]

    """
    Visualization
    """
    # decode the prediction results
    classes = [obj["category"] for obj in objects]
    classes = list(set(classes))
    class_name_to_id = {name: id for id, name in enumerate(classes)}
    class_id_to_name = {id: name for name, id in class_name_to_id.items()}

    boxes = []
    masks = []
    confidences = []
    class_names = []
    class_ids = []

    for idx, obj in enumerate(objects):
        boxes.append(obj["bbox"])
        masks.append(rle_to_array(obj["mask"]["counts"], obj["mask"]["size"][0] * obj["mask"]["size"][1]).reshape(
            obj["mask"]["size"]))
        confidences.append(obj["score"])
        cls_name = obj["category"].lower().strip()
        class_names.append(cls_name)
        class_ids.append(class_name_to_id[cls_name])

    boxes = np.array(boxes)
    masks = np.array(masks)
    class_ids = np.array(class_ids)
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    img = cv2.imread(img_path)
    detections = sv.Detections(
        xyxy=boxes,
        mask=masks.astype(bool),
        class_id=class_ids,
    )

    logging.info(f"save result: {save_name}")
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    cv2.imwrite(os.path.join(OUTPUT_DIR, save_name + "_no_label.jpg"), annotated_frame)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(OUTPUT_DIR, save_name + ".jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(OUTPUT_DIR, save_name + "_with_mask.jpg"), annotated_frame)


def main():
    image_folder_dir = "/data/joyiot/leo/datasets/cam_data_own/"
    image_list = [name for name in os.listdir(image_folder_dir) if name.endswith('.jpg')]
    img_num = len(image_list)
    sample_rate = 80

    for img_index in tqdm(range(img_num)):
        if img_index % sample_rate == 0:
            image_name = "color_{}.jpg".format(img_index)
            img_path = os.path.join(image_folder_dir, image_name)
            sys.stdout.write(f"当前处理: {img_index}")
            sys.stdout.flush()  # 确保立即刷新输出
            infer(img_path=img_path, save_name=str(img_index))


if __name__ == "__main__":
    time_s = time.time()
    main()
    print(f"耗时：{time.time() - time_s}")
