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
        self.roi = (480, 230, 790, 430)

        # 🔥 작은 초록 박스 제거 기준
        self.MIN_GREEN_BOX_AREA = 250   # px^2
        self.MIN_GREEN_BOX_EDGE = 30     # px

        print("✅ ROI + Depth + DBSCAN + Small Green Box Filter Ready")

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

        boxes = []

        for label in set(labels):
            if label == -1:
                continue

            idx = np.where(labels == label)[0]
            pixels = np.array([points_2d[i] for i in idx], dtype=np.float32)

            if len(pixels) < 20:
                continue

            rect = cv2.minAreaRect(pixels)
            (w_rect, h_rect) = rect[1]

            # 🔥 너무 작은 초록 박스 제거
            if w_rect * h_rect < self.MIN_GREEN_BOX_AREA:
                continue
            if min(w_rect, h_rect) < self.MIN_GREEN_BOX_EDGE:
                continue

            box = cv2.boxPoints(rect).astype(int)
            boxes.append(box)

        return boxes

    # -------------------------------
    # RGB + Depth Discontinuity → Compact Transparent Box
    # -------------------------------
    def extract_transparent_rgb(self):
        gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)

        bright = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31, -5
        )

        edges1 = cv2.Canny(gray, 20, 60)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)

        rgb_mask = cv2.bitwise_or(bright, edges)

        depth = self.depth.astype(np.float32)
        depth_blur = cv2.medianBlur(depth, 5)
        grad = np.abs(cv2.Laplacian(depth_blur, cv2.CV_32F))

        depth_hole = np.zeros_like(depth, dtype=np.uint8)
        depth_hole[(depth == 0) | (grad > 20)] = 255

        mask = cv2.bitwise_and(rgb_mask, depth_hole)

        # ROI 제한
        x1, y1, x2, y2 = self.roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        mask = cv2.erode(mask, np.ones((1, 1), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        roi_area = (x2 - x1) * (y2 - y1)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < roi_area * 0.05:
                continue

            cnt_mask = np.zeros_like(mask)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

            dist = cv2.distanceTransform(cnt_mask, cv2.DIST_L2, 5)
            _, _, _, max_loc = cv2.minMaxLoc(dist)
            cx, cy = max_loc

            box_size = int(np.sqrt(area) * 0.6)

            x1b = max(cx - box_size // 2, 0)
            y1b = max(cy - box_size // 2, 0)
            x2b = min(cx + box_size // 2, mask.shape[1])
            y2b = min(cy + box_size // 2, mask.shape[0])

            boxes.append((x1b, y1b, x2b, y2b))

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

        cv2.imshow("Compact Transparent Detection", vis)
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
