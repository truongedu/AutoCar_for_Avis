import cv2
import numpy as np

def draw_lane_lines(image, left_coeffs, right_coeffs, color= (255, 0, 0), h_start = 0.5 ):
    h, _ = image.shape[:2]
    y_vals = np.linspace(h* h_start, h-1, int(h* (1 - h_start)))
    result_img = image.copy()
    black_img = np.zeros_like(image)
    if left_coeffs is not None and right_coeffs is not None:
        try:
            left_x_fit = np.polyval(left_coeffs, y_vals)
            right_x_fit = np.polyval(right_coeffs, y_vals)
            left_pts = np.array([np.stack([left_x_fit, y_vals], axis=-1)], dtype=np.int32)
            right_pts = np.array([np.flipud(np.stack([right_x_fit, y_vals], axis=-1))], dtype=np.int32)
            # all_pts = np.hstack((left_pts, right_pts))
            cv2.polylines(black_img, left_pts, isClosed=False, color=color, thickness=2)
            cv2.polylines(black_img, right_pts, isClosed=False, color= (0,0,255), thickness=2 )
            # cv2.fillPoly(black_img, all_pts, color)
            result_img = cv2.addWeighted(result_img, 1.0, black_img, 0.8, 0)
        except (TypeError, ValueError):
            print('Error : Invalid Coeffs')
            pass
    return result_img