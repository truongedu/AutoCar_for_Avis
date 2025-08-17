import cv2
import numpy as np

def nothing(x):
    pass

# image_path = 'C:\\Axis_car\\code_axis\\bev_image.jpg' 
image_path = 'C:\\Axis_car\\code_axis\\test_sign.jpg'
cv2.namedWindow('bev_image')
cv2.resizeWindow('bev_image', 640, 480)
cv2.createTrackbar('H_min', 'bev_image', 0, 179, nothing)
cv2.createTrackbar('H_max', 'bev_image', 179, 179, nothing)
cv2.createTrackbar('S_min', 'bev_image', 0, 255, nothing)
cv2.createTrackbar('S_max', 'bev_image', 255, 255, nothing)
cv2.createTrackbar('V_min', 'bev_image', 0, 255, nothing)
cv2.createTrackbar('V_max', 'bev_image', 255, 255, nothing)

img = cv2.imread(image_path)
if img is None:
    print("Error: Could not read image.")
    exit()

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
while True:
    h_min = cv2.getTrackbarPos('H_min', 'bev_image')
    h_max = cv2.getTrackbarPos('H_max', 'bev_image')
    s_min = cv2.getTrackbarPos('S_min', 'bev_image')
    s_max = cv2.getTrackbarPos('S_max', 'bev_image')
    v_min = cv2.getTrackbarPos('V_min', 'bev_image')
    v_max = cv2.getTrackbarPos('V_max', 'bev_image')

    lower_= np.array([h_min, s_min, v_min]) # 0, 0, 0
    upper_= np.array([h_max, s_max, v_max]) # 50, 210, 93

    mask = cv2.inRange(hsv_img, lower_, upper_)
    cv2.imshow('bev_image', mask)
    cv2.imshow('Original Image', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


######## White Lane HSV Range :
# Lower: [0, 0, 163]
# Upper: [70, 90, 255]

######## Nau Lane HSV Range :
# Lower: [0, 95, 110]  
# Upper: [75, 210, 255]

######## Blue traffic sign HSV Range :
# Lower: [90, 80, 60]
# Upper: [130, 255, 225]