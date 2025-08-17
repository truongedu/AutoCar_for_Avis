import cv2
import numpy as np
import collections

class Line_Notepad():
    def __init__(self, n_frames=8,threshold_missed = 15 ):
        self.detected = False
        self.recent_fits = collections.deque(maxlen=n_frames)
        self.current_fit = None # Hệ số đa thức được làm mượt qua N lần fit gần nhất
        self.best_fit = None
        self.frames_missed = 0
        self.threshold_missed = threshold_missed 
        
    def update_fit(self, fit_coeffs):
        """
        """
        if fit_coeffs is not None:
            self.detected = True
            self.frames_missed = 0
            self.recent_fits.append(fit_coeffs)
            self.current_fit = fit_coeffs
            self.best_fit = np.mean(self.recent_fits, axis=0)
        else:
            self.detected = False
            self.frames_missed += 1
            if self.frames_missed > self.threshold_missed:
                self.current_fit = None
                self.best_fit = None
                self.recent_fits.clear()


def polyfit_line(x_points, y_points, degree= 2, flag= 'LEFT'):
    """
    Nội suy một đa thức bậc hai vào các điểm ảnh của làn đường.

    Args:
        y_points (np.array): tọa độ Y của các pixel.
        x_points (np.array): tọa độ X của các pixel.
        degree (int): Bậc của đa thức để nội suy.

    Returns:
        np.array: Một mảng chứa các hệ số của đa thức (ví dụ: [A, B, C])
                  nếu nội suy thành công.
        None: Nếu có lỗi xảy ra (ví dụ: không đủ điểm).
    """
    try:
        return np.polyfit(y_points,x_points, degree)
    except(TypeError, np.linalg.LinAlgError) as e:
        print(f"{flag} : Error in polyfit_line: {e}")
        return None
    