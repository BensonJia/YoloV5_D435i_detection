
将D435深度相机和yolov5结合到一起，在识别物体并计算物体质心在相机坐标系下的坐标
D435i是一个搭载IMU（惯性测量单元，采用的博世BMI055）的深度相机，D435i的2000万像素RGB摄像头和3D传感器可以30帧/秒的速度提供分辨率高达1280 × 720，或者以90帧/秒的速度提供848 × 480的较低分辨率。该摄像头为全局快门，可以处理快速移动物体，室内室外皆可操作。深度距离在0.1 m~10 m之间


**注意：** Python版本不可高于3.10

- 配置运行环境

```bash
pip install -r requirements.txt

pip install pyrealsense2
```

# 程序运行

命令行cd 进入工程文件夹下
```bash
python ObjectDetection.py
```
# 功能介绍

**利用深度图信息获取相机坐标系下三维坐标代码如下**
```python
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
```
原理可参考以下文章，[文章链接](https://blog.csdn.net/qq_39211006/article/details/129621260#:~:text=1.%E5%81%87%E8%AE%BE%E6%B7%B1%E5%BA%A6%E5%9B%BE%E7%9A%84%E5%A4%A7%E5%B0%8F%E4%B8%BAHxW%EF%BC%8C%E8%A6%81%E8%8E%B7%E5%8F%96%E5%83%8F%E7%B4%A0%E7%82%B9%20%28i%2Cj%29%E7%9A%84%E4%B8%89%E7%BB%B4%E5%9D%90%E6%A0%87%202.%E9%A6%96%E5%85%88%EF%BC%8C%E9%9C%80%E8%A6%81%E5%B0%86%E8%AF%A5%E5%83%8F%E7%B4%A0%E7%82%B9%E7%9A%84%E5%9D%90%E6%A0%87%E4%BB%8E%E5%9B%BE%E5%83%8F%E5%9D%90%E6%A0%87%E7%B3%BB%E8%BD%AC%E6%8D%A2%E5%88%B0%20%E7%9B%B8%E6%9C%BA%E5%9D%90%E6%A0%87%E7%B3%BB,%E3%80%82%20%E8%AE%BE%E5%83%8F%E7%B4%A0%E7%82%B9%20%28i%2Cj%29%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%80%BC%E4%B8%BAD%20%28i%2Cj%29%2C%E7%9B%B8%E6%9C%BA%E7%9A%84%E5%86%85%E5%8F%82%20%28fx%2Cfy%2Cu0%2Cv0%29%E5%B7%B2%E7%BB%8F%E9%80%9A%E8%BF%87%E7%9B%B8%E6%9C%BA%E6%A0%87%E5%AE%9A%E5%BE%97%E7%9F%A5%EF%BC%8C%E5%88%99%E8%AF%A5%E5%83%8F%E7%B4%A0%E7%82%B9%E5%9C%A8%E7%9B%B8%E6%9C%BA%E5%9D%90%E6%A0%87%E7%B3%BB%E7%9A%84%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F%E5%A6%82%E4%B8%8B%E5%BC%8F%EF%BC%9A)
****
- **使用自己的模型或权重文件**
修改**ObjectDetection**文件下的参数即可
```python
 # Customed Weights File Path & Model Path
    weights_path = 'yolov5-D435i/yolov5s.pt'   #更换为自己的权重文件路径
    model_path = 'yolov5'   #更换为自己的模型路径
    
    # 修改'device'参数以更改运行设备， 0 为GPU
    model = torch.hub.load(model_path, 'custom', path=weights_path, source='local', device='0')
    model.conf = 0.8 # 配置置信度
```
# 参考项目
[https://github.com/ultralytics/yolov5]()
[https://github.com/mushroom-x/yolov5-simple]()
[https://github.com/Thinkin99/yolov5_d435i_detection]()
