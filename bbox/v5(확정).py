import cv2
import numpy as np
from sklearn.cluster import DBSCAN

COLOR_PATH = "color.jpg"
DEPTH_PATH = "depth.npy"
CALIB_PATH = "camcalib.npz"


class DepthDBSCANVisualizer:
    def __init__(self):
        self.color = cv2.imread(COLOR_PATH)
        self.depth = np.load(DEPTH_PATH)

        if self.color is None or self.depth is None:
            raise RuntimeError("❌ color.png 또는 depth.npy 로드 실패")

        calib = np.load(CALIB_PATH)
        self.T_cam_to_work = calib["T_cam_to_work"]
        self.camera_matrix = calib["camera_matrix"]

        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]

        self.depth_scale = 0.001
        self.Z_floor = -2.0

        self.roi = (460, 150, 830, 490)

        print("✅ ROI + Depth + Transparent Detection 준비 완료")

    def in_roi(self, u, v):
        x1, y1, x2, y2 = self.roi
        return x1 <= u <= x2 and y1 <= v <= y2

    def pixel_to_world(self, u, v, depth_cm):
        Xc = (u - self.cx) * depth_cm / self.fx
        Yc = (v - self.cy) * depth_cm / self.fy
        Zc = depth_cm - 1.4
        Pc = np.array([Xc, Yc, Zc, 1.0])
        Pw = self.T_cam_to_work @ Pc
        return Pw[:3]

    # -------------------------------
    # Depth → DBSCAN → box
    # -------------------------------
    def extract_objects_dbscan(self, eps=2.0, min_samples=50):
        h, w = self.depth.shape
        points_2d = []
        points_world = []

        for v in range(0, h, 2):
            for u in range(0, w, 2):
                if not self.in_roi(u, v):
                    continue

                d = self.depth[v, u]
                if d <= 0:
                    continue

                depth_cm = d * self.depth_scale * 100.0
                Pw = self.pixel_to_world(u, v, depth_cm)

                if Pw[2] <= self.Z_floor:
                    points_2d.append((u, v))
                    points_world.append(Pw)

        if len(points_world) == 0:
            return []

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_world)
        labels = clustering.labels_

        boxes = []
        for label in set(labels):
            if label == -1:
                continue

            idx = np.where(labels == label)[0]
            pixels = np.array([points_2d[i] for i in idx])

            x1, y1 = pixels.min(axis=0)
            x2, y2 = pixels.max(axis=0)

            boxes.append((int(x1), int(y1), int(x2), int(y2)))

        return boxes

    # -------------------------------
    # RGB → Transparent box
    # -------------------------------
    def extract_transparent_rgb(self):
        gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        _, bright = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(gray, 30, 100)

        mask = cv2.bitwise_or(bright, edges)
        depth_hole = (self.depth == 0).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, depth_hole)

        x1, y1, x2, y2 = self.roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, x + w, y + h))

        return boxes

    # -------------------------------
    # 실행
    # -------------------------------
    def run(self):
        vis = self.color.copy()

        # ROI
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 초록 박스 (불투명)
        green_boxes = self.extract_objects_dbscan()

        for i, box in enumerate(green_boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 파랑 박스 (투명)
        blue_boxes = self.extract_transparent_rgb()

        # 🔥 겹침 제거
        blue_boxes = suppress_blue_boxes(blue_boxes, green_boxes)

        for i, box in enumerate(blue_boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("ROI + Depth + Transparent Objects (Box-level)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def suppress_blue_boxes(blue_boxes, green_boxes):
    filtered = []

    for bx1, by1, bx2, by2 in blue_boxes:
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2

        keep = True
        for gx1, gy1, gx2, gy2 in green_boxes:
            if gx1 <= cx <= gx2 and gy1 <= cy <= gy2:
                keep = False
                break

        if keep:
            filtered.append((bx1, by1, bx2, by2))

    return filtered


if __name__ == "__main__":
    app = DepthDBSCANVisualizer()
    app.run()
