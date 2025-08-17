import cv2
import numpy as np

point =[]

def mouse_click(event, x,y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        point.append((x,y))
        print(f"Point added: ({x}, {y})")
        cv2.circle(image, (x,y), 5, (0,255,0), -1)
        cv2.imshow("Diem", image)

img_path= 'C:\Axis_car\code_axis\output.jpg'
image = cv2.imread(img_path)
if image is None:
    print("Error: Image not found.")
    exit()

img_cp = image.copy()
h, w = img_cp.shape[:2]
cv2.namedWindow("Diem")
cv2.setMouseCallback("Diem", mouse_click)
cv2.imshow("Diem", img_cp)
while True:
    key = cv2.waitKey(1) & 0xFF
    if len(point) == 4 or key == ord('q'):
        for i, p in enumerate(point):
            print(f" (w * {p[0]/w:.2f} ,h * {p[1]/h:.2f}),")
        break
 
cv2.destroyAllWindows()