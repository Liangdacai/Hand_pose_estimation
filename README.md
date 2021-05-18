This Demo is based on yolov5s for hand detection, and lightweight-human-pose-estimation for hand 21 keypoints estimation
yolov5 github:https://github.com/ultralytics/yolov5
lightweight-human-pose-estimation:https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch

At the end of the hand keypoints network, a few layers of network are added to directly return to 21 hand keypoints

How to run the algorithm

Note:The weight of checkpoint_iter_1100000.pth does not include the structure of the direct return to 21 hand keypoints, 
only the heatmap can be calculated

run demo
--python detect.py

if you want to use heatmap to calculate hand keypoints,--use_heatmap==True

https://github.com/Liangdacai/Hand_pose_estimation/blob/Jay-Neo/inference/aaa.jpg?raw=true

if you want to directly return to 21 hand keypoints,--use_heatmap==True
