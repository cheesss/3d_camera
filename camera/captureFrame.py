import threading
import cv2
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from PIL import Image

def save_RGB_img(image):
    # RGB 이미지를 JPEG 파일로 저장
    cv2.imwrite("captured_image.jpg", image)
    print("RGB image saved as 'captured_image.jpg'")


def save_Depth_img(depth_image, depth_scale):
    depth_image_normalized = (depth_image / depth_scale).astype(np.float32)
    # depth_image_normalized = np.clip(depth_image_normalized, 0, 1)  # 클리핑

    # PIL로 변환 후 저장
    depth_image_pil = Image.fromarray(depth_image_normalized)
    depth_image_pil.save("captured_depth_image.png")
    print("Depth image saved as 'captured_depth_image.png'")


def capture_save_show_image():
    pipeline = rs.pipeline()
    config = rs.config()


    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale: {depth_scale} meters per unit")
    align = rs.align(rs.stream.color)

    try:
        pipeline.start(config)
        # 프레임 읽기
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("No frames captured!")
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        print(np.average(depth_image))
    
        # RGB 및 깊이 이미지를 저장하는 스레드 시작
        threading.Thread(target=save_RGB_img, args=(color_image,)).start()
        threading.Thread(target=save_Depth_img, args=(depth_image, depth_scale)).start()

        cv2.imshow('Captured RGB Image', color_image)
        cv2.imshow('Captured Depth Image', (depth_image / depth_scale).astype(np.float32))

        # ESC 키로 종료
        while True:
            key = cv2.waitKey(1)
            if key == 27:  # ESC 키
                break
            if cv2.getWindowProperty('Captured RGB Image', cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        pipeline.stop()
        print("Pipeline stopped, resources released.")

# 실행
capture_save_show_image()
