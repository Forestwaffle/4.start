import realsense_viewer
import cam_event

# RealSense 카메라 실행
realsense_viewer.run(
    on_save=cam_event.Save_Cam,            # 스페이스바 누르면 color 이미지와 depth 데이터를 저장
    on_reset=cam_event.reset_points,       # 'r' 키 누르면 클릭된 좌표를 모두 초기화
    on_click=cam_event.mouse_callback,     # 마우스 좌클릭 이벤트를 처리, 클릭한 좌표와 depth값 저장
    update_depth_frame=cam_event.update_depth_frame,   # 매 프레임의 depth 정보를 cam_event 전역 변수로 전달
    update_color_image=cam_event.update_color_image,   # 매 프레임의 color 이미지를 cam_event 전역 변수로 전달
    get_points=cam_event.get_saved_points, # 현재 클릭된 좌표 목록을 반환, 화면에 빨간 점으로 표시
)
