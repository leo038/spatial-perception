import os

import cv2
from tqdm import tqdm  # python 进度条库

image_folder_dir = "/data/joyiot/leo/datasets/cam_data_own/"
image_folder_dir = "/data/joyiot/leo/codes/spatial-perception/outputs/yolo11_det/"
image_folder_dir = "/data/joyiot/liyong/codes/GroundingDINO/outputs/groundingdino_det/"
fps = 1  # fps: frame per leo 每秒帧数，数值可根据需要进行调整
size = (1280, 720)  # (width, height) 数值可根据需要进行调整
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 编码为 mp4v 格式，注意此处字母为小写，大写会报错
video = cv2.VideoWriter("./result_gdino.mp4", fourcc, fps, size, isColor=True)

image_list = [name for name in os.listdir(image_folder_dir) if
              name.endswith('.jpg')]  # 获取文件夹下所有格式为 jpg 图像的图像名，并按时间戳进行排序
images_num = len(image_list)

save_interval = 80

for index in tqdm(range(8000)):  # 遍历 image_list 中所有图像并添加进度条
    if index % save_interval == 0:
        image_name = "{}.jpg".format(index)
        image_full_path = os.path.join(image_folder_dir, image_name)  # 获取图像的全路经
        image = cv2.imread(image_full_path)  # 读取图像
        video.write(image)  # 将图像写入视频

video.release()
cv2.destroyAllWindows()
