# Copyright (c) Tencent Inc. All rights reserved.
import argparse
import os
import os.path as osp
from copy import deepcopy
from io import BytesIO

import cv2
import numpy as np
import onnx
import onnxsim
import supervision as sv
import torch
from PIL import Image
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms

from yolo_world.easydeploy.model import DeployModel, MMYOLOBackend

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)


def parse_args():
    parser = argparse.ArgumentParser(
        description='YOLO-World Demo')
    parser.add_argument('--config',
                        default='/data/joyiot/leo/codes/YOLO-World/configs/pretrain/yolo_world_xl_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py',
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        default='/data/joyiot/leo/codes/YOLO-World/checkpoints/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain.pth',
                        help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def run_image(runner,
              image,
              text,
              max_num_boxes,
              score_thr,
              nms_thr,
              image_path='./work_dirs/demo.png'):
    os.makedirs('./work_dirs', exist_ok=True)
    # image.save(image_path)
    cv2.imwrite(image_path, image)
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"保存检测结果")
    cv2.imwrite("./reuslt.jpg", image)

    # cv2.namedWindow('yolo-world', flags=cv2.WINDOW_NORMAL |
    #                                        cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    #
    # cv2.imshow('yolo-world', color_image)
    # cv2.waitKey(0)

    image = Image.fromarray(image)
    return image


def export_model(runner=None,
                 checkpoint=None,
                 text=None,
                 max_num_boxes=100,
                 score_thr=0.05,
                 nms_thr=0.5):
    backend = MMYOLOBackend.ONNXRUNTIME
    postprocess_cfg = ConfigDict(
        pre_top_k=10 * max_num_boxes,
        keep_top_k=max_num_boxes,
        iou_threshold=nms_thr,
        score_threshold=score_thr)

    base_model = deepcopy(runner.model)
    texts = [[t.strip() for t in text.split(',')] + [' ']]
    base_model.reparameterize(texts)
    deploy_model = DeployModel(
        baseModel=base_model,
        backend=backend,
        postprocess_cfg=postprocess_cfg)
    deploy_model.eval()

    device = (next(iter(base_model.parameters()))).device
    # fake_input = torch.ones([1, 3, 640, 640], device=device)
    fake_input = torch.ones([1, 3, 1280, 1280], device=device)
    # dry run
    deploy_model(fake_input)

    os.makedirs('work_dirs', exist_ok=True)
    save_onnx_path = os.path.join(
        'work_dirs', 'yolow-l.onnx')
    # export onnx
    with BytesIO() as f:
        output_names = ['num_dets', 'boxes', 'scores', 'labels']
        torch.onnx.export(
            deploy_model,
            fake_input,
            f,
            input_names=['images'],
            output_names=output_names,
            opset_version=12)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
    onnx_model, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, save_onnx_path)


def demo(runner, args, cfg):
    prompt = "person, desk, sofa, chair,monitor,  lamp, plant, door,cup, bottle, window, mouse, keyboard, cabinet, bin, latch,drawer, flag, handbag, box, pillow,flowerpot, picture, trophy, speaker, shelf, monitor, arm, flower,blackboard,sign,mirror,trolley,book,router,text,toy,couch,fan,table, refrigerator, light, camera, telephone,power outlet, carpet, curtain, hinge,glasses,shoes"

    img_file = '/data/joyiot/leo/datasets/cam_data_own/color_4960.jpg'
    image = cv2.imread(img_file)
    run_image(runner=runner, image=image, text=prompt, max_num_boxes=100, score_thr=0.05,
              nms_thr=0.5)  ## 注意， 一定要完整推理一次再导出， 否则推理可以导出， 但推理会报错

    export_model(runner=runner, checkpoint=args.checkpoint, text=prompt, max_num_boxes=100, score_thr=0.05, nms_thr=0.5)


if __name__ == '__main__':
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    os.makedirs('./work_dirs', exist_ok=True)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    runner.model.eval()
    demo(runner, args, cfg)
