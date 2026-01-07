import cv2
import numpy as np

# ===============================
# 파일 경로
# ===============================
COLOR_PATH = "color.png"
DEPTH_PATH = "depth.npy"
CALIB_PATH = "camcalib.npz"


class TransparentObjectVisualizer:
    def __init__(self):
        # -------------------------------
        # 데이터 로드
        # -------------------------------
        self.color = cv2.imread(COLOR_PATH)
        self.depth = np.load(DEPTH_PATH)  # mm

        if self.color is None or self.depth is None:
            raise RuntimeError("❌ color.png 또는 depth.npy 로드 실패")

        # ROI (x1, y1, x2, y2)
        self.roi = (460, 150, 830, 490)

        print("✅ ROI + RGB + Depth-hole 기반 투명 물체 검출 준비 완료")

    # -------------------------------
    # ROI 내부 여부
    # -------------------------------
    def apply_roi_mask(self, mask):
        x1, y1, x2, y2 = self.roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255
        return cv2.bitwise_and(mask, roi_mask)

    # -------------------------------
    # RGB → Transparent Object
    # -------------------------------
    def extract_transparent_rgb(self):
        img = self.color.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1️⃣ Brightness (검정 배경 가정)
        _, bright = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # 2️⃣ Edge
        edges = cv2.Canny(gray, 30, 100)

        # 3️⃣ Bright + Edge
        mask = cv2.bitwise_or(bright, edges)

        # 4️⃣ Depth hole (투명 물체 핵심)
        depth_hole = (self.depth == 0).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, depth_hole)

        # 5️⃣ ROI 적용
        mask = self.apply_roi_mask(mask)

        # 6️⃣ Morphology
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 7️⃣ Contour
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        objects = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            objects.append((x, y, x + w, y + h))

        return objects

    # -------------------------------
    # 실행
    # -------------------------------
    def run(self):
        vis = self.color.copy()

        # ROI 시각화
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, "ROI", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 투명 물체 검출
        t_objects = self.extract_transparent_rgb()

        for i, (x1, y1, x2, y2) in enumerate(t_objects):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis, f"transparent_{i}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("ROI + Transparent Objects", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = TransparentObjectVisualizer()
    app.run()
