import os
import sys
import time
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("./checkpoints/yolo11x.pt")

OUTPUT_DIR = Path("./outputs/yolo11_det/")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def infer(img_path, save_name):
    results = model(img_path)
    # results[0].show()
    save_name = os.path.join(OUTPUT_DIR, save_name +'.jpg')
    results[0].save(save_name)


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
    # time_s = time.time()
    # main()
    # print(f"耗时：{time.time() - time_s}")

    infer(img_path="/data/joyiot/leo/datasets/cam_data_own/color_2294.jpg", save_name="test")
