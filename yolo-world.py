import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv
from PIL import Image, ImageOps
from tqdm import tqdm

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()

OUTPUT_DIR = Path("./outputs/yolo11-world/")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

text_prompt = "person, desk, sofa, chair,monitor,  lamp, plant, door,cup, bottle, window, mouse, keyboard, cabinet, bin, latch,drawer, flag, handbag, box, pillow,flowerpot, picture, trophy, speaker, shelf, monitor, arm, flower,blackboard,sign,mirror,trolley,book,router,text,toy,couch,fan,table, refrigerator, light, camera, telephone,power outlet, carpet, curtain, hinge,glasses,shoes"
texts = [[t.strip()] for t in text_prompt.split(',')] + [[' ']]
print(f"texts:{texts}")


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    # Get sample input data as a numpy array in a method of your choosing.
    img_width, img_height = image.size
    size = max(img_width, img_height)
    image = ImageOps.pad(image, (size, size), method=Image.BILINEAR)
    image = image.resize((1280, 1280), Image.BILINEAR)
    tensor_image = np.asarray(image).astype(np.float32)
    tensor_image /= 255.0
    tensor_image = np.expand_dims(tensor_image, axis=0)
    return tensor_image, (img_width, img_height, size)


def visualize(results, img):
    bboxes = results[2][0]
    scores = results[1][0]
    labels = results[0][0]
    bboxes = bboxes[labels >= 0]
    scores = scores[labels >= 0]
    labels = labels[labels >= 0]

    print(bboxes.shape)
    detections = sv.Detections(xyxy=bboxes, class_id=labels, confidence=scores)
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    # label images
    image = (img * 255).astype(np.uint8)
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    return image


def infer(ort_session=None, img_path='/data/joyiot/leo/datasets/cam_data_own/color_4960.jpg', save_name=None):
    # ort_session = ort.InferenceSession(onnx_file_name)
    img, meta_info = load_image(img_path)
    input_ort = ort.OrtValue.ortvalue_from_numpy(img.transpose((0, 3, 1, 2)))
    time_s = time.time()
    results = ort_session.run(["labels", "scores", "boxes"], {"images": input_ort})
    print(f"帧率：{1 / (time.time() - time_s)}")
    img_out = visualize(results, img[0])

    save_name = os.path.join(OUTPUT_DIR, save_name + '.jpg')
    print(f"保存检测结果:{save_name}")
    cv2.imwrite(save_name, img_out)


def main():
    image_folder_dir = "/data/joyiot/leo/datasets/cam_data_own/"
    image_list = [name for name in os.listdir(image_folder_dir) if name.endswith('.jpg')]

    img_num = len(image_list)
    sample_rate = 1

    onnx_file_name = "./work_dirs/yolow-l.onnx"
    ort_session = ort.InferenceSession(onnx_file_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    for img_index in tqdm(range(img_num)):
        if img_index % sample_rate == 0:
            image_name = "color_{}.jpg".format(img_index)
            img_path = os.path.join(image_folder_dir, image_name)
            infer(img_path=img_path, save_name=str(img_index), ort_session=ort_session)


if __name__ == "__main__":
    # time_s = time.time()
    # main()
    # print(f"耗时：{time.time() - time_s}")

    onnx_file_name = "./work_dirs/yolow-l.onnx"
    ort_session = ort.InferenceSession(onnx_file_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    infer(img_path="./color_2294.jpg", save_name="test", ort_session=ort_session)
