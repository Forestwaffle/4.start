import pyrealsense2 as rs
import numpy as np
import cv2
import os

SAVE_DIR = "."
os.makedirs(SAVE_DIR, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)

print("▶ 's' : 저장 | 'q' : 종료")

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        cv2.imshow("Color", color)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(f"{SAVE_DIR}/color.jpg", color)
            np.save(f"{SAVE_DIR}/depth.npy", depth)
            print("✅ 저장 완료: color.png / depth.npy")

        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()