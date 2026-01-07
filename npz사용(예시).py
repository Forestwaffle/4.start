import cv2
import numpy as np
import pyrealsense2 as rs
import os

# ===============================
# 설정 및 파일 경로
# ===============================
SAVE_FILE = "camcalib.npz"

class RealsenseCoordinatePicker:
    def __init__(self):
        # 1. RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # 2. 저장된 캘리브레이션 데이터 로드
        self.load_calibration()

    def load_calibration(self):
        """저장된 변환 행렬 및 카메라 파라미터 로드"""
        if not os.path.exists(SAVE_FILE):
            print(f"❌ '{SAVE_FILE}' 파일을 찾을 수 없습니다. 먼저 캘리브레이션 코드를 실행하세요.")
            exit()
        
        data = np.load(SAVE_FILE)
        self.T_cam_to_work = data["T_cam_to_work"]
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]
        
        print(f"✅ '{SAVE_FILE}' 로드 완료 (저장된 캘리브레이션 사용)")
        print(f"   - fx: {self.camera_matrix[0,0]:.2f}, fy: {self.camera_matrix[1,1]:.2f}")
        print(f"   - cx: {self.camera_matrix[0,2]:.2f}, cy: {self.camera_matrix[1,2]:.2f}\n")

    def pixel_to_world(self, u, v, depth_frame):
        """픽셀 좌표를 월드 좌표로 변환"""
        # 클릭 지점 주변 Median Depth 추출 (0 제외 및 노이즈 제거)
        depth_list = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                d = depth_frame.get_distance(u + i, v + j)
                if d > 0: 
                    depth_list.append(d)
        
        if not depth_list:
            return None
        
        # 미터 단위 depth를 cm로 변환
        depth_cm = np.median(depth_list) * 100.0

        # 카메라 좌표계(Pc)에서의 3D 좌표 계산
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        
        Xc = (u - cx) * depth_cm / fx
        Yc = (v - cy) * depth_cm / fy
        Zc = depth_cm - 1.4
        
        Pc = np.array([Xc, Yc, Zc, 1.0])  # 동차 좌표계 (Homogeneous)

        # 월드 좌표계 변환 (Pw = T * Pc)
        Pw = self.T_cam_to_work @ Pc
        return Pw[:3]

if __name__ == "__main__":
    # 사용 예시
    app = RealsenseCoordinatePicker()
    
    print("[초기화 완료]")
    print("이제 pixel_to_world(u, v, depth_frame) 메서드를 사용할 수 있습니다.\n")
    
    # 예시: 프레임 받아서 좌표 변환
    try:
        frames = app.pipeline.wait_for_frames()
        aligned = app.align.process(frames)
        depth_f = aligned.get_depth_frame()
        
        # 화면 중앙 좌표 테스트
        u, v = 640, 360
        world_pos = app.pixel_to_world(u, v, depth_f)
        if world_pos is not None:
            print(f"테스트: 픽셀({u},{v}) -> 월드 X={world_pos[0]:.2f}, Y={world_pos[1]:.2f}, Z={world_pos[2]:.2f} cm")
    finally:
        app.pipeline.stop()