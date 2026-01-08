import cv2
from realsense_manager import RealSenseManager
from cam_event import (setup_click_collector,update_depth_frame,update_color_image,Save_Cam ,reset_points,get_saved_points)
import rclpy
from std_msgs.msg import Float32MultiArray
from coordinatetr import Coordinate

def main():
    # ROS 초기화
    rclpy.init()
    node = rclpy.create_node('garbage_sender')
    pub = node.create_publisher(Float32MultiArray, 'garbage_topic', 10)
    sent_space = False

    # RealSense 초기화
    cam = RealSenseManager()
    cv2.namedWindow("RealSense Color")
    setup_click_collector("RealSense Color")

    # 좌표 변환 클래스
    picker = Coordinate()

    print("카메라 시작")
    print("클릭: 포인트 추가")
    print("s: 이미지 저장 | r: 리셋 | SPACE: 발행 | t: 발행 리셋 | esc: 종료")

    while True:
        color_image, depth_frame = cam.get_frames()
        if color_image is None or depth_frame is None:
            continue

        # click_collector에 프레임 전달 (중요)
        update_depth_frame(depth_frame)
        update_color_image(color_image)

        # 클릭 포인트 시각화
        display_image = color_image.copy()
        for x, y, _ in get_saved_points():
            cv2.circle(display_image, (int(x), int(y)), 5, (0, 0, 255), -1)

        cv2.imshow("RealSense Color", display_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'): #사진 저장
            Save_Cam()

        elif key == ord('r'): #리셋
            reset_points()

        elif key == ord(' '): #스페이스 토픽보내기
            if not sent_space:
                world_points = []

                for x, y, _ in get_saved_points():
                    Pw = picker.pixel_to_world(x, y, depth_frame)
                    if Pw is not None:
                        world_points.append([Pw[0], Pw[1], Pw[2]])

                flat_points = [float(v) for p in world_points for v in p]

                msg = Float32MultiArray()
                msg.data = flat_points
                pub.publish(msg)

                sent_space = True
                print(world_points)

        elif key == ord('t'): #토픽 횟수 초기화
            sent_space = False

        elif key == 27: #esc
            break

    cam.stop()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
