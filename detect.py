import argparse
import os
import shutil
import cv2
import torch
from models.experimental import attempt_load

from utils.general import (
    check_img_size, non_max_suppression, scale_coords, plot_one_box, set_logging)
from utils.torch_utils import select_device, time_synchronized
import numpy as np

from hand_pose_model.with_mobilenet import PoseEstimationWithMobileNet
from hand_pose_model.conv import load_state


BODY_PARTS_KPT_IDS =[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
                      [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[1,5],[5,9],[9,13],[13,17]]

KNOWN_DISTANCE = 50.0
KNOWN_WIDTH = 5.0
focalLength = (56 * KNOWN_DISTANCE) / KNOWN_WIDTH
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth



def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def detect():
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    #----------------hand_pose---------#
    hand_pose = PoseEstimationWithMobileNet(opt.use_heatmap,3)
    checkpoints = torch.load(opt.hand_pose_weights, map_location='cpu')
    load_state(hand_pose, checkpoints)
    hand_pose = hand_pose.eval()
    hand_pose = hand_pose.cuda()

    cap = cv2.VideoCapture(0)
    import numpy as np

    while cap.isOpened():

        _, frame = cap.read()
        # img = letterbox(frame, new_shape=320, auto=True)[0]
        img = cv2.resize(frame, (320, 256))

        img = np.array([img])
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    # ------------------hand--pose---------#
                    pad = 30
                    c1, c2 = (int(xyxy[0] - pad), int(xyxy[1] - pad)), (int(xyxy[2] + pad), int(xyxy[3] + pad))
                    crop = frame[c1[1]:c2[1], c1[0]:c2[0]]
                    h, w, _ = crop.shape

                    if h == 0 or w == 0:
                        continue
                    crop = cv2.resize(crop, (256, 256))
                    pose_inputs = normalize(crop, np.array([128, 128, 128], np.float32), np.float32(1 / 256))
                    pose_inputs = torch.from_numpy(pose_inputs).permute(2, 0, 1).unsqueeze(0).float().cuda()
                    output = hand_pose(pose_inputs)
                    output = output.cpu().data.numpy()

                    keypoints = []


                    for i in range(21):
                        if opt.use_heatmap:
                            temp = output[0, i, :, :]
                            xy = np.where(temp == np.max(temp))
                            cv2.circle(frame,(int(xy[1][0] * 4 * w / 256 + c1[0]), int(xy[0][0] * 4 * h / 256 + c1[1])), 3,
                                       (0, 0, 0), 3)
                            keypoints.append([int(xy[1][0] * 4 * w / 256 + c1[0]), int(xy[0][0] * 4 * h / 256 + c1[1])])
                        else:
                            cv2.circle(frame, (int(output[0][2 * i] * 224 * w / 256 + c1[0]),
                                            int(output[0][2 * i + 1] * 224 * h / 256) + c1[1]), 3, (0, 0, 0), 3)
                            keypoints.append([int(output[0][2 * i] * 224 * w / 256 + c1[0]),
                                              int(output[0][2 * i + 1] * 224 * h / 256) + c1[1]])



                    for line in BODY_PARTS_KPT_IDS:
                        cv2.line(frame, (keypoints[line[0]][0], keypoints[line[0]][1]),
                                 (keypoints[line[1]][0], keypoints[line[1]][1]), (0, 255, 0), 2)

                    dis_01 = np.sqrt(
                        (keypoints[0][0] - keypoints[1][0]) ** 2 + (keypoints[0][1] - keypoints[1][1]) ** 2)
                    dis = distance_to_camera(KNOWN_WIDTH, focalLength, dis_01)

                    # ------------------hand--pose---------#

                    label = '%s %.2f' % ('camera distace', dis)
                    plot_one_box(xyxy, frame, label=label, color=(0, 0, 255), line_thickness=3)

            cv2.imwrite('./inference/bbb.jpg',frame)
            cv2.imshow("PF_Hand_Pose_Estimate", frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/hand_detect.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models',default=False)
    parser.add_argument('--hand_pose_weights', type=str, default='./hand_pose_model/checkpoint_iter_105000.pth', help='hand pose weights')
    parser.add_argument('--use_heatmap', type=bool, default=False, help='if True use heatmap ,else use 21 points')

    opt = parser.parse_args()

    with torch.no_grad():
        detect()
