import cv2
import numpy as np

def bev_transform(frame):
    """
    Transform the input frame to Bird's Eye View (BEV) perspective.
    Args:
        frame (numpy.ndarray): Input image frame.
    Returns:
        bev_img (numpy.ndarray): Transformed BEV image.
        img_cp (numpy.ndarray): Copy of the input image with the source polygon drawn.
        M_inv (numpy.ndarray): Inverse transformation matrix.
    """
    h,w = frame.shape[:2]
    src_pts = np.float32([
        (w * 0.35 ,h * 0.42),
        (w * 0.635 ,h * 0.42),
        (w * 1.00 ,h * 0.73),
        (w * 0.00 ,h * 0.73)
    ])

    img_cp = frame.copy()
    cv2.polylines(img_cp,[np.int32(src_pts)], isClosed=True, color=(0,0,255), thickness=2)
    # cv2.imshow('Source', img_cp)
    dst_pts = np.float32([
        (0, 0),                # Điểm trên-trái của ảnh đầu ra
        (w, 0),                # Điểm trên-phải của ảnh đầu ra
        (w, h),                # Điểm dưới-phải của ảnh đầu ra
        (0, h)                 # Điểm dưới-trái của ảnh đầu ra
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    bev_img =cv2.warpPerspective(frame, M, (w, h))

    return bev_img, img_cp, M_inv




















# src_pts = np.float32([
#     (0,h* 0.75),
#     (w,h* 0.75),
#     (w*0.64, h*0.4),
#     (w*0.36, h*0.4)
# ])
#     src_pts = np.float32([
#  (w * 0.35 ,h * 0.40),
#  (w * 0.65 ,h * 0.40),
#  (w * 1.00 ,h * 0.76),
#  (w * 0.00 ,h * 0.76),
#     ])

#     src_pts = np.float32([
#  (w * 0.26 ,h * 0.38),
#  (w * 0.43 ,h * 0.38),
#  (w * 0.32 ,h * 0.51),
#  (w * 0.00 ,h * 0.51)
#     ])

    # src_pts = np.float32([
    #     (w * 0.21, h * 0.45),  # Điểm trên-trái
    #     (w * 0.79, h * 0.45),  # Điểm trên-phải
    #     (w * 1.00, h * 1.00),  # Điểm dưới-phải
    #     (w * 0.00, h * 1.00)   # Điểm dưới-trái
    # ])
    

    # offset_x = w * 0.15
    # dst_pts = np.float32([
    #     [offset_x, 0],         # Điểm trên-trái của ảnh đầu ra
    #     [w - offset_x, 0],     # Điểm trên-phải của ảnh đầu ra
    #     [w - offset_x, h],     # Điểm dưới-phải của ảnh đầu ra
    #     [offset_x, h]          # Điểm dưới-trái của ảnh đầu ra
    # ])