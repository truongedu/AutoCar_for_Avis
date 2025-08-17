import cv2
import numpy as np
from scipy.signal import find_peaks
import collections

def detect_lane(image):
    """
    Trả về (error_px, out_img)
    error_px < 0  => xe lệch TRÁI
    error_px > 0  => xe lệch PHẢI
    """
    h, w = image.shape[:2]

    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray', gray)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask  = np.zeros_like(edges)
    poly  = np.array([[(0, h),
                       (w, h),
                       (w*0.8, int(h*0.4)),
                       (w* 0.1, int(h*0.4))]], np.int32)
    cv2.fillPoly(mask, poly, 255)
    roi = cv2.bitwise_and(edges, mask)
    img_cp= image.copy()
    cv2.polylines(img_cp, poly, isClosed=True, color= [0,0,255],thickness=2)
    roi_img = cv2.addWeighted(img_cp, 0.3, image, 0.7 , 0)
    # cv2.imshow('ROI',roi_img)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180,
                            threshold=50,
                            minLineLength=40,
                            maxLineGap=100)

    left_pts, right_pts = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.5:    
                left_pts  += [(x1, y1), (x2, y2)]
            elif slope > 0.5:  
                right_pts += [(x1, y1), (x2, y2)]

    lane_img = image.copy()

    cx = w // 2  # tâm ảnh
    if left_pts and right_pts:
        lx, ly = zip(*left_pts)
        rx, ry = zip(*right_pts)
        left_fit  = np.polyfit(ly, lx, 1)   # x = m*y + b
        right_fit = np.polyfit(ry, rx, 1)

        x_left  = left_fit[0]*h + left_fit[1]
        x_right = right_fit[0]*h + right_fit[1]
        cx      = int((x_left + x_right) / 2)

        cv2.line(lane_img, (int(x_left), h),  (int(x_left - left_fit[0]*h*0.4), int(h*0.6)),  (255,0,0),  3)
        cv2.line(lane_img, (int(x_right), h), (int(x_right - right_fit[0]*h*0.4), int(h*0.6)), (0,255,0), 3)
        cv2.line(lane_img, (cx, h), (cx, int(h*0.6)), (0,255,255), 2)

    error_px = (w // 2) - cx
    return error_px, lane_img, roi 

def detect_edges_from_bev(bev_image, min_area_contour=370):
    white_lower_lane = np.array([0, 0, 163])
    white_upper_lane = np.array([70, 90, 255])
    brown_lower_lane = np.array([0, 95, 110])
    brown_upper_lane = np.array([75, 210, 255])

    hsv_image = cv2.cvtColor(bev_image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv_image, white_lower_lane, white_upper_lane)
    brown_mask = cv2.inRange(hsv_image, brown_lower_lane, brown_upper_lane)
    combined_mask = cv2.bitwise_or(white_mask, brown_mask)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours_mask = np.zeros_like(combined_mask)
    for c in contours:
        c_area = cv2.contourArea(c)
        # print(f"Contour area: {c_area}")
        if c_area > min_area_contour:
            cv2.drawContours(filtered_contours_mask,[c], -1, 255, -1)
    blur_mask = cv2.GaussianBlur(filtered_contours_mask, (5, 5), 0)
    edges = cv2.Canny(blur_mask, 50, 150)
    return edges
    
def detect_edges_from_image(image):
    """
    Phát hiện cạnh từ ảnh đầu vào.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def detect_lines_from_egdes(edges_img, min_line_length=40, max_line_gap=100):
    """
    Phát hiện đường thẳng từ ảnh cạnh.
    """
    lines = cv2.HoughLinesP(edges_img, 1, np.pi/180,
                            threshold=50,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    return lines

def find_lane_start(edges_img):
    '''
    '''
    h = edges_img.shape[0]
    roi_edges = edges_img[int(h* 0.6):, :]
    hist_edges = np.sum(roi_edges, axis = 0)
    # print(hist_edges)
    # print(f'Shape of hist edges:{hist_edges.shape}')
    mid_x_hist = int(hist_edges.shape[0]/2)
    left_x_max_idx = np.argmax(hist_edges[:mid_x_hist])
    right_x_max_idx = np.argmax(hist_edges[mid_x_hist:]) + mid_x_hist
    return left_x_max_idx, right_x_max_idx, hist_edges 

def find_lane_start_new(edges_img, h_threshold=10, min_peak_distance=380):
    h, w = edges_img.shape
    roi_edges = edges_img[int(h * 0.6):, :]
    histogram = np.sum(roi_edges, axis=0)
    image_center_x = w // 2
    all_peaks_indices, _ = find_peaks(histogram, height=h_threshold, distance=min_peak_distance)
    if len(all_peaks_indices) < 2:
        # print("ERROR 1 !")
        return None, None 

    left_peaks = all_peaks_indices[all_peaks_indices < image_center_x]
    right_peaks = all_peaks_indices[all_peaks_indices >= image_center_x]
    if len(left_peaks) == 0 or len(right_peaks) == 0:
        print("ERROR 2 !")
        return None, None
    left_start_x = left_peaks[np.argmin(abs(left_peaks - image_center_x))]
    right_start_x = right_peaks[np.argmin(abs(right_peaks - image_center_x))]
    
    return left_start_x, right_start_x


def find_lane_pixels(edges_img, start_x,
                    n_window = 12,margin_width_window= 16, min_pixels=16
                    , search_height_ratio = 1.0):
    h = edges_img.shape[0]
    height_of_window = h // n_window
    x_current = start_x
    xy_nonzero = np.nonzero(edges_img)
    y_nonzero = np.array(xy_nonzero[0])
    x_nonzero = np.array(xy_nonzero[1])
    lane_idxs = []
    n_windows_to_search = int(n_window * search_height_ratio)
    for i_win in range(n_windows_to_search):
        lower_y_window = h - (i_win+1) * height_of_window
        upper_y_window = h - i_win * height_of_window
        lower_x_window = x_current - margin_width_window
        upper_x_window = x_current + margin_width_window
        # cv2.rectangle(out_img,
        #               (lower_x_window, lower_y_window),
        #               (upper_x_window, upper_y_window),
        #               (255, 0, 0), 2) # Blue

        good_idxs = ((y_nonzero >= lower_y_window) 
                    & (y_nonzero < upper_y_window)
                    & (x_nonzero >= lower_x_window)
                    & (x_nonzero < upper_x_window)).nonzero()[0]
        lane_idxs.append(good_idxs)
        if len(good_idxs)> min_pixels:
            x_current = int(np.mean(x_nonzero[good_idxs]))
    lane_idxs = np.concatenate(lane_idxs)
    x_points = x_nonzero[lane_idxs]
    y_points = y_nonzero[lane_idxs]
    # out_img[y_points, x_points] = [0, 0, 255] # Red
    return x_points, y_points


def search_around_poly(edges_img, fit_coeffs_odd, margin = 20, h_start = 0.5):
    '''
      h_start : 0.5
    '''
    h,_ = edges_img.shape[:2]
    roi_start = int(h * h_start)
    roi_edges = edges_img[roi_start:, :]
    xy_nonezero = np.nonzero(roi_edges)
    y_roi = np.array(xy_nonezero[0])
    x_roi = np.array(xy_nonezero[1])
    y_full_coords = y_roi + roi_start   
    # fit_lane_x = fit_coeffs_odd[0] * y_full_coords**2 + fit_coeffs_odd[1] * y_full_coords + fit_coeffs_odd[2]
    fit_lane_x = np.polyval(fit_coeffs_odd, y_full_coords)
    good_idxs = ((x_roi >= (fit_lane_x - margin)) &
                 (x_roi <= (fit_lane_x + margin)))
    x_points = x_roi[good_idxs]
    y_points = y_full_coords[good_idxs] 
    return x_points, y_points






















def find_lane_pixels_odd(edges_img, left_start_x, right_start_x, 
                     n_windows=12, margin_width_window= 16, min_pixels=16,
                     search_height_ratio = 1.0):
    """
    """
    # out_img = np.dstack((edges_img, edges_img, edges_img)) * 255
    h = edges_img.shape[0]
    height_of_window = h // n_windows
    left_x_current = left_start_x
    right_x_current = right_start_x
    xy_nonzero = np.nonzero(edges_img)
    y_nonzero = np.array(xy_nonzero[0])
    x_nonzero = np.array(xy_nonzero[1])
    left_lane_idxs = []
    right_lane_idxs = []
    n_windows_to_search = int(n_windows * search_height_ratio)
    for i_win in range(n_windows_to_search):
        lower_y_window = h - (i_win+1) * height_of_window
        upper_y_window = h - i_win * height_of_window
        left_lower_x_window = left_x_current - margin_width_window
        left_upper_x_window = left_x_current + margin_width_window
        right_lower_x_window = right_x_current - margin_width_window
        right_upper_x_window = right_x_current + margin_width_window
        # cv2.rectangle(out_img,
        #               (left_lower_x_window, lower_y_window),
        #               (left_upper_x_window, upper_y_window),
        #               (255, 0, 0), 2) # Blue
        # cv2.rectangle(out_img,
        #               (right_lower_x_window, lower_y_window),
        #               (right_upper_x_window, upper_y_window),
        #               (0, 0, 255), 2) # Red
        good_left_idxs = ((y_nonzero >= lower_y_window) 
                        & (y_nonzero < upper_y_window)
                        & (x_nonzero >= left_lower_x_window)
                        & (x_nonzero < left_upper_x_window)).nonzero()[0]
        good_right_idxs = ((y_nonzero >= lower_y_window) 
                         & (y_nonzero < upper_y_window)
                         & (x_nonzero >= right_lower_x_window)
                         & (x_nonzero < right_upper_x_window)).nonzero()[0]
        left_lane_idxs.append(good_left_idxs)
        right_lane_idxs.append(good_right_idxs)
        if len(good_left_idxs)> min_pixels:
            left_x_current = int(np.mean(x_nonzero[good_left_idxs]))
        if len(good_right_idxs)> min_pixels:
            right_x_current = int(np.mean(x_nonzero[good_right_idxs]))
    left_lane_idxs = np.concatenate(left_lane_idxs)
    right_lane_idxs = np.concatenate(right_lane_idxs)
    left_x = x_nonzero[left_lane_idxs]
    left_y = y_nonzero[left_lane_idxs]
    right_x = x_nonzero[right_lane_idxs]
    right_y = y_nonzero[right_lane_idxs]
    # out_img[left_y, left_x] = [0, 0, 255] # Red
    # out_img[right_y, right_x] = [255, 0, 0] # Blue
    return left_x, left_y, right_x, right_y