##############################################################
#   camera.py
#   version: 3.2.2 (edited in 2023.08.21)
##############################################################
import sys
import os
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import yaml

import pyrealsense2 as rs
import numpy as np
import open3d as o3d

import copy

class IntelCamera:
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.context = rs.context()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.spatial_filter = rs.spatial_filter()
        self.hole_filling_filter = rs.hole_filling_filter(0)

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

        print(self.device_product_line + " is ready")
        self.device_name = device.get_info(rs.camera_info.name).replace(" ", "_")
        self.device_name = self.device_name + "_" + device.get_info(rs.camera_info.serial_number)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.color_intrinsic = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intrinsic = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.fx = self.color_intrinsic.fx
        self.fy = self.color_intrinsic.fy
        self.ppx = self.color_intrinsic.ppx
        self.ppy = self.color_intrinsic.ppy

        if self.device_product_line == 'L500':
            self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(960, 540, self.fx, self.fy, self.ppx, self.ppy)
        
        else:
            self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(640, 480, self.fx, self.fy, self.ppx, self.ppy)

        self.camera_mat = np.array([[self.fx, 0, self.ppx], [0, self.fy, self.ppy], [0, 0, 1]], dtype=np.float64)

        self.dist_coeffs = np.zeros(4)
        self.colorizer = rs.colorizer(color_scheme = 2)

        self.saw_yaml = False
        self.z_min = -0.05
    
    def stream(self):
        
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        ## filter depth frame
        # depth_frame = self.spatial_filter.process(depth_frame)
        # depth_frame = self.hole_filling_filter.process(depth_frame)

        colored_depth_frame = self.colorizer.colorize(depth_frame)

        self.color_image = np.asanyarray(color_frame.get_data())
        self.colored_depth_image = np.asanyarray(colored_depth_frame.get_data())
        self.depth_image = np.asanyarray(depth_frame.get_data())

        return self.color_image, self.depth_image

    def generate(self, depth, downsample=True):
        depth_o3d = o3d.geometry.Image(depth)
        if self.device_product_line == 'L500':
            self.pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, self.intrinsic_o3d, depth_scale=4000.0)
        else:
            self.pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, self.intrinsic_o3d, depth_scale=1000.0)
        if downsample:
            self.pcd = self.pcd.voxel_down_sample(voxel_size = 0.006)
        self.xyz = np.asarray(self.pcd.points)
        return self.xyz

if __name__ == '__main__':

    # cam = IntelCamera(cfg=[])
    # cam.create_aruco_marker(aruco.DICT_6X6_50)

    # with open(ref_path+"/core/config/suction_config.yml") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    #     cfg['TF']['end2cam'] = np.reshape(cfg['TF']['end2cam'], (4, 4))
    cfg = []
    # cam = KinectCamera(cfg)
    cam = IntelCamera(cfg)

    while 1:
        rgb_img, depth_img = cam.stream()
        # cam.detectAruco()

        # print(cam.cam2marker)
        # print(np.average(depth_img*0.00025))
        # xyz = cam.generate(depth_img)
        # cam.detectCharuco()
        # xyz = cam.cropPoints()

        ## visualize rgb and depth image
        cv2.imshow("rgb", rgb_img)
        # cv2.imshow("depth", depth_img)
        cv2.waitKey(1)