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

        # 바닥 높이 (Z 기준)
        self.Z_floor = -2.0

        print("✅ Depth + RGB Transparent Detection 준비 완료")

    # -------------------------------
    # Pixel → World
    # -------------------------------
    def pixel_to_world(self, u, v, depth_cm):
        Xc = (u - self.cx) * depth_cm / self.fx
        Yc = (v - self.cy) * depth_cm / self.fy
        Zc = depth_cm - 1.4  # 카메라 높이 보정

        Pc = np.array([Xc, Yc, Zc, 1.0])
        Pw = self.T_cam_to_work @ Pc
        return Pw[:3]

    # -------------------------------
    # Depth → DBSCAN (불투명 물체)
    # -------------------------------
    def extract_objects_dbscan(
        self,
        Z_threshold=None,
        eps=1.0,
        min_samples=50
    ):
        if Z_threshold is None:
            Z_threshold = self.Z_floor

        h, w = self.depth.shape
        points_2d = []
        points_world = []

        # Down-sampling
        for v in range(0, h, 2):
            for u in range(0, w, 2):
                d = self.depth[v, u]
                if d <= 0:
                    continue

                depth_cm = d * self.depth_scale * 100.0
                Pw = self.pixel_to_world(u, v, depth_cm)

                # 바닥 위 포인트만
                if Pw[2] <= Z_threshold:
                    points_2d.append([u, v])
                    points_world.append(Pw)

        if len(points_world) == 0:
            return []

        points_world = np.array(points_world)

        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples
        ).fit(points_world)

        labels = clustering.labels_
        objects = []

        for label in set(labels):
            if label == -1:
                continue

            idx = np.where(labels == label)[0]
            pixels = np.array([points_2d[i] for i in idx])

            x1, y1 = pixels.min(axis=0)
            x2, y2 = pixels.max(axis=0)

            objects.append((int(x1), int(y1), int(x2), int(y2)))

        return objects

    # -------------------------------
    # RGB → Transparent Object
    # -------------------------------
    def extract_transparent_rgb(self):
        img = self.color.copy()

        # Gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Brightness (검정 배경 기준)
        _, bright = cv2.threshold(
            gray,
            15,
            255,
            cv2.THRESH_BINARY
        )

        # Edge
        edges = cv2.Canny(gray, 30, 100)

        # 결합
        mask = cv2.bitwise_or(bright, edges)

        # Depth hole만 남김 (핵심)
        depth_hole = (self.depth == 0).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, depth_hole)

        # Morphology
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Contour → bbox
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            objects.append((x, y, x + w, y + h))

        return objects

    # -------------------------------
    # 실행
    # -------------------------------
    def run(self):
        vis = self.color.copy()

        # 1. 불투명 물체 (Depth)
        objects = self.extract_objects_dbscan(
            Z_threshold=self.Z_floor,
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

        # 2. 투명 물체 (RGB)
        t_objects = self.extract_transparent_rgb()

        for i, (x1, y1, x2, y2) in enumerate(t_objects):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                vis,
                f"transparent_{i}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        cv2.imshow("Depth + Transparent Objects", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = DepthDBSCANVisualizer()
    app.run()
