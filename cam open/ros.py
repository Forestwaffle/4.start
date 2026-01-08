import cv2
from realsense_manager import RealSenseManager
from click_collector import (
    setup_click_collector,
    update_depth_frame,
    print_points,
    reset_points,
    get_saved_points
)

import rclpy
from std_msgs.msg import Float32MultiArray

# 좌표 변환 모듈 import
from coordinatetr import Coordinate

def main():
    # ROS 초기화 및 퍼블리셔 생성
    rclpy.init()
    node = rclpy.create_node('garbage_sender')
    pub = node.create_publisher(Float32MultiArray, 'garbage_topic', 10)
    sent_space = False  # space 한 번만 발행 제어

    # RealSense 초기화
    cam = RealSenseManager()
    cv2.namedWindow("RealSense Color")
    setup_click_collector("RealSense Color")

    # 좌표 변환 클래스 초기화
    picker = Coordinate()

    print("RealSense 카메라가 켜졌습니다.")
    print("마우스 클릭: 포인트 추가")
    print("s: 클릭 확인 | r: 리셋 | SPACE: 좌표 발행 | t: 발행 리셋 | q: 종료")

    while True:
        color_image, depth_frame = cam.get_frames()
        if color_image is None or depth_frame is None:
            continue

        update_depth_frame(depth_frame)
        cv2.imshow("RealSense Color", color_image)

        key = cv2.waitKey(1) & 0xFF

        # 클릭 포인트 확인 / 리셋
        if key == ord('s'):
            print_points()
        elif key == ord('r'):
            reset_points()

        # SPACE: 클릭 좌표 리스트 발행 (한 번만)
        elif key == ord(' '):
            if not sent_space:
                points = get_saved_points()  # 클릭 픽셀 좌표 가져오기
                world_points = []

                # 픽셀 좌표를 월드 좌표로 변환
                for u, v in points:
                    Pw = picker.pixel_to_world(u, v, depth_frame)
                    if Pw is not None:
                        world_points.append([Pw[0], Pw[1], Pw[2]])

                # 2차원 월드 좌표를 1차원 float 배열로 flatten
                flat_points = [float(c) for p in world_points for c in p]

                msg = Float32MultiArray()
                msg.data = flat_points
                pub.publish(msg)

                sent_space = True
                print(f"월드 좌표 발행 완료: {world_points}")

        # t: 발행 리셋
        elif key == ord('t'):
            sent_space = False
            print("발행 상태 리셋 완료")

        # 종료
        elif key == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()
    print("카메라 종료")

    # ROS 종료
    node.destroy_node()
    rclpy.shutdown()
    print("ROS 종료 완료")

if __name__ == "__main__":
    main()
