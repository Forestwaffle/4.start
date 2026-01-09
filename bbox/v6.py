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

        # ROI (u1, v1, u2, v2)
        self.roi = (473, 220, 800, 436)

        print("✅ ROI + Depth + DBSCAN(회전 박스) + Transparent 준비 완료")

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

                # 바닥 위의 물체
                if Pw[2] <= self.Z_floor:
                    points_2d.append([u, v])
                    points_world.append(Pw)

        if len(points_world) == 0:
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
            # rect = (center, (w, h), angle)
            box = cv2.boxPoints(rect)
            box = box.astype(int)

            rotated_boxes.append(box)

        return rotated_boxes

    # -------------------------------
    # RGB → Transparent box (AABB)
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

        # ROI 표시
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 🟩 회전 박스 (불투명 물체)
        green_rotated_boxes = self.extract_objects_dbscan_rotated()

        for box in green_rotated_boxes:
            cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)

        # 🟦 투명 물체 (축 정렬 박스)
        blue_boxes = self.extract_transparent_rgb()

        blue_boxes = suppress_blue_boxes(blue_boxes, green_rotated_boxes)

        for bx1, by1, bx2, by2 in blue_boxes:
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

        cv2.imshow("ROI + Depth + DBSCAN Rotated Box + Transparent", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# -------------------------------
# 초록 박스 내부 파랑 박스 제거
# -------------------------------
def suppress_blue_boxes(blue_boxes, green_rotated_boxes):
    filtered = []

    for bx1, by1, bx2, by2 in blue_boxes:
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2

        keep = True
        for box in green_rotated_boxes:
            if cv2.pointPolygonTest(box, (cx, cy), False) >= 0:
                keep = False
                break

        if keep:
            filtered.append((bx1, by1, bx2, by2))

    return filtered


if __name__ == "__main__":
    app = DepthDBSCANVisualizer()
    app.run()
