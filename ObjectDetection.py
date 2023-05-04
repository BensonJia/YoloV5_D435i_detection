import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import yaml
import json
import random
import time

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
pipeline.start(config)

def get_pos(frame, box, depth_frame):
    
    mid_pos = ((box[0] + box[2])/2.0, (box[1] + box[3])/2.0)
    mid_depth = depth_frame.get_distance(int(mid_pos[0]), int(mid_pos[1]))
    
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    '''
    camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }
    '''
    
    camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (mid_pos[0], mid_pos[1]), mid_depth)  # 计算相机坐标系的xyz
    camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
    camera_xyz = camera_xyz.tolist()

    return mid_pos, mid_depth, camera_xyz

def dectshow(org_image, boxs, depth_frame):
    image = org_image.copy()
    print("=================================")
    for box in boxs:
        #color = [np.random.randint(0,255) for _ in range(3)]
        color = (0, 255, 255)
        pos1, pos2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, pos1, pos2, color, 2)
        mid_pos, mid_depth, pos_xyz = get_pos(org_image, box, depth_frame)
        mid_pos = (int(mid_pos[0]), int(mid_pos[1]))
        cv2.circle(image, (int(mid_pos[0]), int(mid_pos[1])), 4, (0, 255, 0), -1)   # mark mid_pos of this object
        cv2.putText(image, box[-1] + " conf: " + str(box[-3])[:5] , pos1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, str(pos_xyz), mid_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        print("obj{}[{}] conf: {}".format(box[-2],box[-1],box[-3]))
    cv2.imshow("Detection", image)

if __name__ == '__main__':

    print("[INFO] Program Start")
    print("[INFO] Loading YOLOv5 model")
    # Customed Weights File Path & Model Path
    weights_path = 'yolov5-D435i/yolov5s.pt'
    model_path = 'yolov5'

    model = torch.hub.load(model_path, 'custom', path=weights_path, source='local', device='0')
    model.conf = 0.8
    
    print("[INFO] Loaded Successfully")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            t_start = time.time()
            results = model(color_image)
            t_end = time.time()
            fps = int(1.0 / (t_end - t_start))
            cv2.putText(color_image, text="FPS: {}".format(fps), org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                        lineType=cv2.LINE_AA, color=(0, 0, 0))

            boxs = results.pandas().xyxy[0].values
            dectshow(color_image, boxs, depth_frame)

            key = cv2.waitKey(1)
    finally:
        pipeline.stop()