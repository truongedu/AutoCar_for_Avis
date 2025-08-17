import avisengine
import cv2
import numpy as np
import time
from detect_transform_for_lines.line_detect import *
from detect_transform_for_lines.BEV_transform import bev_transform
from detect_transform_for_lines.line_logic import *
from Visualize_image import draw_lane_lines
from cal_steering import PIDController
from detect_traffic_signs import detect_traffic_signs

SIMULATOR_IP   = "127.0.0.1"
SIMULATOR_PORT = 25001

DESIRED_SPEED  = 15                   
MAX_THROTTLE   = 20
MIN_THROTTLE   = 0
KP_SPEED       = 3.0  
KP_STEER       = 0.25   
KI             = 0.002
KD             = 0.05       
MAX_STEER      = 25           
LANE_WIDTH_PIXELS = 430

car = avisengine.Car()
car.connect(SIMULATOR_IP, SIMULATOR_PORT)
time.sleep(3)
car.setSensorAngle(30)           
left_line_note = Line_Notepad(n_frames=8, threshold_missed=15)
right_line_note = Line_Notepad(n_frames=8, threshold_missed=15)
cal_steering = PIDController(kp=KP_STEER, ki=KI, kd=KD, setpoint=0)
# model_detect_line = cv2.dnn.readNetFromONNX('C:\Axis_car\code_axis\V1.0\traffic_sign_classifier_lenet_v3.onnx')

try:
    while True:
        car.getData()
        frame = car.getImage()
        curent_speed = car.getSpeed()
        steer_cmd = 0
        if frame is not None:
            bev, _, _ =bev_transform(frame)  # Apply BEV transformation
            edges_img = detect_edges_from_bev(bev)  # Detect edges from BEV image
            lx, ly, rx, ry = [], [], [], []
            if left_line_note.best_fit is not None:
                lx, ly = search_around_poly(edges_img, left_line_note.best_fit, h_start=0.0)
            if right_line_note.best_fit is not None:
                rx, ry = search_around_poly(edges_img, right_line_note.best_fit)    
            
            if not np.any(lx) or not np.any(rx):
                xleft_start, xright_start = find_lane_start_new(edges_img)
                if xleft_start is not None and not np.any(lx):
                    lx, ly= find_lane_pixels(edges_img, xleft_start,  search_height_ratio=1.0)
                if xright_start is not None and not np.any(rx):
                    rx, ry = find_lane_pixels(edges_img, xright_start, search_height_ratio=0.5)
          
            left_fit_coeffs = polyfit_line(lx, ly, degree=2)
            right_fit_coeffs = polyfit_line(rx, ry, degree=2, flag='RIGHT')
            left_line_note.update_fit(left_fit_coeffs)
            right_line_note.update_fit(right_fit_coeffs)
            stable_left_coeffs = left_line_note.best_fit
            stable_right_coeffs = right_line_note.best_fit
            
            if not left_line_note.detected and right_line_note.detected and stable_right_coeffs is not None:
                stable_left_coeffs = stable_right_coeffs.copy()
                stable_left_coeffs[2] -= LANE_WIDTH_PIXELS
            elif not right_line_note.detected and left_line_note.detected and stable_left_coeffs is not None:
                stable_right_coeffs = stable_left_coeffs.copy()
                stable_right_coeffs[2] += LANE_WIDTH_PIXELS

            lane_img = draw_lane_lines(bev, stable_left_coeffs, stable_right_coeffs, h_start=0.5)
            if stable_left_coeffs is not None and stable_right_coeffs is not None:
                h, w = lane_img.shape[:2]
                y_values = h -1
                left_x = np.polyval(stable_left_coeffs, y_values)
                right_x = np.polyval(stable_right_coeffs, y_values)
                cv2.circle(lane_img, (int(left_x), h - 1), 5, (255, 0, 0), -1)  # Blue : Left Lane
                cv2.circle(lane_img, (int(right_x), h - 1), 5, (0, 0, 255), -1) # Red : Right Lane
                center_lane_x = (left_x + right_x) / 2
                if center_lane_x<0 or center_lane_x > w:
                    print(f"LANE_WIDTH :{center_lane_x} because of {left_x} and {right_x}")
                error =   w / 2 - center_lane_x
                steer_cmd = cal_steering.update_steering(error)
                steer_cmd = np.clip(steer_cmd, -MAX_STEER, MAX_STEER)

                cv2.circle(lane_img, (int(center_lane_x), h - 1), 5, (0, 255, 0), -1) # Green : Tam Lane
                cv2.circle(lane_img, (int(w/2), h-1), 5, (255, 255, 0), -1) # Xanh da troi: Tam Car
                cv2.putText(lane_img, f'Steering: {steer_cmd:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Original", frame)
            # cv2.imshow("BEV", bev)
            # cv2.imshow("ROI", img_roi)
            # cv2.imshow("Edges", edges_img)
            # cv2.imshow("Lane Lines", lane_img)
            # cv2.imshow("left_right_lane_pixels", vis_img_right)

        # --- Setup SPEED and STEERING ---
        throttle = KP_SPEED * (DESIRED_SPEED - curent_speed)
        throttle = np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE)
        car.setSpeed(throttle)
        # print(f' Steetring: {steer_cmd: .2f}')
        car.setSteering(steer_cmd)

        # --- Visualization ---
        

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite('test_sign.jpg', frame)
            print("Saved image as test_sign.jpg")

        time.sleep(0.02)  

finally:
    car.stop()
    cv2.destroyAllWindows()
    print("Ngắt kết nối")

