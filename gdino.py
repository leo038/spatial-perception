import os
import sys
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from groundingdino.util.inference import load_model, load_image, predict, annotate
from prompt import PROMPT

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model = load_model("groundingdino/config/GroundingDINO_SwinB_cfg.py",
                   "checkpoints/groundingdino_swinb_cogcoor.pth").cuda()
TEXT_PROMPT = PROMPT
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
OUTPUT_DIR = Path("./outputs/groundingdino_det/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def infer(img_path, save_name):
    image_source, image = load_image(img_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    save_name = os.path.join(OUTPUT_DIR, save_name + ".jpg")
    cv2.imwrite(save_name, annotated_frame)


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
