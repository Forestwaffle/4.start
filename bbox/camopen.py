import pyrealsense2 as rs
import numpy as np
import cv2

# 1. 파이프라인 생성
pipeline = rs.pipeline()
config = rs.config()

# 2. Color + Depth 스트림 설정 (Depth 켜기만, 화면 출력 X)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# 3. 카메라 켜기
pipeline.start(config)
print("✅ RealSense 카메라가 켜졌습니다. 'q'를 누르면 종료됩니다.")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())

    # 화면에 Color 영상만 표시
    cv2.imshow('RealSense Color', color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
print("카메라 종료")
