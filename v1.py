import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# ===============================
# 파일 경로
# ===============================
COLOR_PATH = "color.png"
DEPTH_PATH = "depth.npy"
CALIB_PATH = "camcalib.npz"


class DepthDBSCANVisualizer:
    def __init__(self):
        # 데이터 로드
        self.color = cv2.imread(COLOR_PATH)
        self.depth = np.load(DEPTH_PATH)  # mm

        if self.color is None or self.depth is None:
            raise RuntimeError("❌ color.png 또는 depth.npy 로드 실패")

        # 캘리브레이션 로드
        calib = np.load(CALIB_PATH)
        self.T_cam_to_work = calib["T_cam_to_work"]
        self.camera_matrix = calib["camera_matrix"]
        self.dist_coeffs = calib["dist_coeffs"]

        # camera_matrix에서 fx, fy, cx, cy 추출
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]

        # 깊이 단위 변환
        self.depth_scale = 0.001  # mm → m

        # 바닥 높이 (Z 기준, 음수 방향)
        self.Z_floor = -2.0  # 바닥 근처 값, 환경에 맞게 조정

        print("✅ Depth + DBSCAN 준비 완료")

    # -------------------------------
    # Pixel → World
    # -------------------------------
    def pixel_to_world(self, u, v, depth_cm):
        # 카메라 좌표
        Xc = (u - self.cx) * depth_cm / self.fx
        Yc = (v - self.cy) * depth_cm / self.fy
        Zc = depth_cm -1.4
        Pc = np.array([Xc, Yc, Zc, 1.0])
        # 월드 좌표 변환
        Pw = self.T_cam_to_work @ Pc
        return Pw[:3]

    # -------------------------------
    # Depth → DBSCAN
    # -------------------------------
    def extract_objects_dbscan(
        self,
        Z_threshold=None,  # 바닥 위 기준
        eps=1.0,
        min_samples=50
    ):
        if Z_threshold is None:
            Z_threshold = self.Z_floor  # 바닥 근처 값

        h, w = self.depth.shape
        points_2d = []
        points_world = []

        # 유효한 depth 포인트 수집 (다운샘플링)
        for v in range(0, h, 2):
            for u in range(0, w, 2):
                d = self.depth[v, u]
                if d <= 0:
                    continue

                depth_cm = d * self.depth_scale * 100.0
                Pw = self.pixel_to_world(u, v, depth_cm)

                # 바닥 위 포인트만 선택 (Z가 음수 방향으로 커질수록 높은 물체)
                if Pw[2] <= Z_threshold:
                    points_2d.append([u, v])
                    points_world.append(Pw)

        if len(points_world) == 0:
            return []

        points_world = np.array(points_world)

        # DBSCAN 클러스터링
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples
        ).fit(points_world)

        labels = clustering.labels_
        objects = []

        for label in set(labels):
            if label == -1:
                continue  # noise

            idx = np.where(labels == label)[0]
            pixels = np.array([points_2d[i] for i in idx])

            x1, y1 = pixels.min(axis=0)
            x2, y2 = pixels.max(axis=0)

            objects.append((int(x1), int(y1), int(x2), int(y2)))

        return objects

    # -------------------------------
    # 실행
    # -------------------------------
    def run(self):
        vis = self.color.copy()

        objects = self.extract_objects_dbscan(
            Z_threshold=self.Z_floor,  # 바닥 위 포인트만
            eps=2.0,
            min_samples=50
        )

        for i, (x1, y1, x2, y2) in enumerate(objects):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"obj_{i}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        cv2.imshow("Depth DBSCAN Objects", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = DepthDBSCANVisualizer()
    app.run()
