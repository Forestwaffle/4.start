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
            raise RuntimeError("❌ color.jpg 또는 depth.npy 로드 실패")

        calib = np.load(CALIB_PATH)
        self.T_cam_to_work = calib["T_cam_to_work"]
        self.camera_matrix = calib["camera_matrix"]

        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]

        self.depth_scale = 0.001
        self.Z_floor = -2.0

        # ROI (u1, v1, u2, v2)
        self.roi = (488, 230, 790, 420)

        # 🔥 하드코딩 파라미터
        self.MIN_VALID_DEPTH_CM = 5.0        # depth 최소값
        self.MIN_BLUE_AREA_RATIO = 0.01      # ROI 대비 파랑 박스 최소 면적 비율

        print("✅ Depth + DBSCAN + Transparent Detection Ready")

    # -------------------------------
    # Utils
    # -------------------------------
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
    # Depth → DBSCAN → Rotated Box
    # -------------------------------
    def extract_objects_dbscan_rotated(self, eps=2.0, min_samples=50):
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
                    points_2d.append([u, v])
                    points_world.append(Pw)

        if not points_world:
            return []

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_world)
        labels = clustering.labels_

        rotated_boxes = []

        for label in set(labels):
            if label == -1:
                continue

            idx = np.where(labels == label)[0]
            pixels = np.array([points_2d[i] for i in idx], dtype=np.float32)

            if len(pixels) < 20:
                continue

            rect = cv2.minAreaRect(pixels)
            box = cv2.boxPoints(rect).astype(int)
            rotated_boxes.append(box)

        return rotated_boxes

    # -------------------------------
    # RGB + Depth → Transparent
    # -------------------------------
    def extract_transparent_rgb(self):
        gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        bright = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31, -5
        )

        edges = cv2.Canny(gray, 30, 120)
        rgb_mask = cv2.bitwise_or(bright, edges)

        depth = self.depth.astype(np.float32)
        depth_blur = cv2.medianBlur(depth, 5)
        grad = np.abs(cv2.Laplacian(depth_blur, cv2.CV_32F))

        depth_hole = np.zeros_like(depth, dtype=np.uint8)
        depth_hole[(depth == 0) | (grad > 20)] = 255

        mask = cv2.bitwise_and(rgb_mask, depth_hole)

        x1, y1, x2, y2 = self.roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        roi_area = (x2 - x1) * (y2 - y1)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # 🔵 너무 작은 파랑 박스 제거 (하드코딩)
            if area < roi_area * self.MIN_BLUE_AREA_RATIO:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi_depth = depth[y:y+h, x:x+w]

            # 1️⃣ Depth 존재 비율
            valid_ratio = np.count_nonzero(roi_depth) / (w * h)
            if valid_ratio < 0.1:
                continue

            # 2️⃣ Depth 최소값
            valid_depth = roi_depth[roi_depth > 0]
            if len(valid_depth) == 0:
                continue

            min_depth_cm = np.min(valid_depth) * self.depth_scale * 100.0
            if min_depth_cm < self.MIN_VALID_DEPTH_CM:
                continue

            boxes.append((x, y, x + w, y + h))

        return boxes

    # -------------------------------
    # 실행
    # -------------------------------
    def run(self):
        vis = self.color.copy()

        x1, y1, x2, y2 = self.roi
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        green_boxes = self.extract_objects_dbscan_rotated()
        for box in green_boxes:
            cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)

        blue_boxes = self.extract_transparent_rgb()
        blue_boxes = suppress_blue_boxes(blue_boxes, green_boxes)

        for bx1, by1, bx2, by2 in blue_boxes:
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

        cv2.imshow("Depth-based Transparent Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# -------------------------------
# 초록 박스 내부 파랑 박스 제거
# -------------------------------
def suppress_blue_boxes(blue_boxes, green_boxes):
    filtered = []

    for bx1, by1, bx2, by2 in blue_boxes:
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2

        keep = True
        for box in green_boxes:
            if cv2.pointPolygonTest(box, (cx, cy), False) >= 0:
                keep = False
                break

        if keep:
            filtered.append((bx1, by1, bx2, by2))

    return filtered


if __name__ == "__main__":
    app = DepthDBSCANVisualizer()
    app.run()
