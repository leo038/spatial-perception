import json
from copy import deepcopy

import cv2
import dashscope
import numpy as np
import supervision as sv

from api_key import API_TOKEN_QWEN
from prompt import PROMPT

dashscope.api_key = API_TOKEN_QWEN

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()

MODEL_NAME = 'qwen2.5-vl-72b-instruct'

print(f"使用的模型：{MODEL_NAME}")
class_names = PROMPT.replace('.', ',')
text_prompt = """用矩形框定位图像中物体的位置， 要定位的物体包括： person, desk, sofa, chair,monitor,  lamp, plant, door,cup, bottle, window, mouse, keyboard, cabinet, bin, latch,drawer, flag, handbag, box, pillow,flowerpot, picture, trophy, speaker, shelf, monitor, arm, flower,blackboard,sign,mirror,trolley,book,router,text,toy,couch,fan,table, refrigerator, light, camera, telephone,power outlet, carpet, curtain, hinge,glasses,shoes
，以JSON格式输出所有的bbox的坐标，不要输出```json```代码段。输出格式为： [{"label": "object_name","bboxes": [xmin, ymin, xmax, ymax]}, ...], 请严格按照给定的输出格式要求， 一个label只对应一个bboxes. label 也务必在给定的范围内。"""

print(f"text_prompt:{text_prompt}")

texts = [[t.strip()] for t in class_names.split(',')] + [[' ']]

texts_tmp = [t.strip() for t in class_names.split(',')]
class_id_dict = {}
for idx, class_name in enumerate(texts_tmp):
    class_id_dict.update({class_name: idx})

print(f"class_id_dict:{class_id_dict}")


def infer(img_path):
    messages = [{
        'role': 'user',
        'content': [
            {
                'image': img_path
            },
            {
                'text': text_prompt
            },
        ]
    }]

    response = dashscope.MultiModalConversation.call(model=MODEL_NAME, messages=messages)
    print(f"原始response:{response}")

    result = response.output.choices[0].message.content[0]['text']
    ## 注意这里可能会出错， 因为模型有时候指令遵循不是很好， 输入了一些无效内容或括号不匹配， 比如```json ```, ```josn等等， 3B, 7B, 32B 基本上每次都出现。
    res_json = parse_result(result)

    return res_json


def parse_result(result):
    try:
        res_json = json.loads(result)
    except:
        print("模型输出了```json等无效内容")
        if "```json" in result:
            result = result[7:]
        if "```" in result:
            result = result[:-3]
        if result.endswith(','):
            result = result[:-1]  # 去掉末尾的逗号
        try:
            res_json = json.loads(result)

        except:
            print("模型输出内容括号不匹配")
            try:
                res_tmp = deepcopy(result)
                res_tmp += "}]"
                res_json = json.loads(res_tmp)
            except:
                try:
                    res_tmp = deepcopy(result)
                    res_tmp += "]"
                    res_json = json.loads(res_tmp)
                except:
                    res_tmp = deepcopy(result)
                    res_tmp += "]}]"
                    res_json = json.loads(res_tmp)

    print(f"json格式化后结果:{res_json}")
    return res_json


def _format_result(json_result):
    """大模型返回的数据格式如下：
    [
    {"label": "person", "bboxes": [54, 336, 227, 710]},
    {"label": "desk", "bboxes": [218, 449, 297, 520]},
    {"label": "sofa", "bboxes": [0, 529, 93, 714]},
    {"label": "chair", "bboxes": [218, 413, 284, 454]},
    {"label": "monitor", "bboxes": [294, 376, 316, 399]},
    {"label": "lamp", "bboxes": [820, 137, 857, 200]},
    {"label": "plant", "bboxes": [199, 305, 547, 728]},
    {"label": "cup", "bboxes": [668, 424, 688, 451]},
    {"label": "flowerpot", "bboxes": [373, 669, 477, 728]},
    {"label": "trophy", "bboxes": [831, 349, 892, 460]},
    {"label": "shelf", "bboxes": [480, 330, 1283, 728]},
    {"label": "arm", "bboxes": [69, 438, 103, 535]},
    {"label": "light", "bboxes": [820, 137, 857, 200]},
    {"label": "shoes", "bboxes": [124, 662, 163, 712]} ]

    """
    bboxes = []
    class_names = []
    for det in json_result:
        try:
            label = det['label']
        except:
            print(f"label名字错误， 实际为labels")
            label = det.get('labels')

        try:
            bbox = det['bboxes']
        except:
            print(f"bboxes名字错误， 实际为bbox")
            bbox = det.get('bbox')

        if label is None or bbox is None:
            continue

        bboxes.append(bbox)
        class_names.append(label)

    bboxes = np.array(bboxes)
    class_names = np.array(class_names)
    print(f"检测的目标数量：{len(class_names)}")

    class_ids = [class_id_dict[name] for name in class_names]
    class_ids = np.array(class_ids)

    return bboxes, class_ids


def visualize(results, image):
    bboxes, labels = _format_result(json_result=results)
    # scores = np.ones(len(labels))
    detections = sv.Detections(xyxy=bboxes, class_id=labels)
    labels = [f"{texts[class_id][0]} " for class_id in detections.class_id]

    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    return image


if __name__ == "__main__":
    img_path = "/data/joyiot/leo/datasets/cam_data_own/color_4960.jpg"

    res_json = infer(img_path)

    image = cv2.imread(img_path)
    out_image = visualize(res_json, image)

    save_name = MODEL_NAME + "_4960.jpg"
    print(f"保存输出图像:{save_name}")
    cv2.imwrite(save_name, out_image)
