import cv2
import numpy as np

def filter_signs_by_color(frame):
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    # mask_red1 = cv2.inRange(hsv_img, lower1, upper1)

    lower3, upper3 = np.array([90, 80, 60]), np.array([130, 255, 225])
    mask_b = cv2.inRange(hsv_img, lower3,upper3)
    mask_red1 = mask_b.copy()
    mask_final = cv2.bitwise_or(mask_red1, mask_b)
    cv2.imshow('Mask', mask_final)
    return mask_final

def get_bbox_from_mask(mask):
    h_mask, w_mask = mask.shape[:2]
    n_components, labels_seg, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    bboxes = []
    # print(f"Number of components: {n_components}")
    for i in range(1,n_components):
        x = stats[i,cv2.CC_STAT_LEFT]
        y= stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w < 20 or h < 20:
            continue
        if w > 0.8 * w_mask or h > 0.8 * h_mask:
            continue
        # Loại bỏ các vật có tỷ lệ dài / rộng quá khác biệt
        if w / h > 2.0 or h / w > 2.0:
            continue
        bboxes.append((x, y, w, h))
    return bboxes

def detect_traffic_signs(frame, model, draw = False):
    classes = ['unknown', 'left', 'no_left', 'right',
               'no_right', 'straight', 'stop']
    mask = filter_signs_by_color(frame)
    bboxes = get_bbox_from_mask(mask)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_norm = frame.astype(np.float32) / 255.0
    signs = []
    # print(bboxes)
    for bbox in bboxes:
        print(bbox)
        x,y,w,h = bbox
        sub_img = frame_norm[y:y+h, x:x+w]
        sub_img =cv2.resize(sub_img, (32,32))
        cv2.imshow('Sub Image', sub_img)
        sub_img = np.expand_dims(sub_img, axis=0)
        model.setInput(sub_img)
        preds = model.forward()
        preds = preds[0]
        cls = preds.argmax()
        score = preds[cls]
        print(f"Class: {classes[cls]}, Score: {score:.2f}, BBox: {bbox}")
        # if cls ==0 or score <0.8:
        #     continue
        signs.append([classes[cls], score, x, y, w, h])
        
        if draw:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
            text = f'{classes[cls]}-{score:.2f}'
            cv2.putText(frame, text, (x-20, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return signs, frame


if __name__ == '__main__':
    model_detect_line = cv2.dnn.readNetFromONNX('traffic_sign_classifier_lenet_v2.onnx')
    img_test = cv2.imread('test_sign.jpg')
    signs, img_with_bbox_signs = detect_traffic_signs(img_test, model_detect_line, draw = True)
    print("Detected signs:", signs)
    img_with_bbox_signs = cv2.cvtColor(img_with_bbox_signs, cv2.COLOR_RGB2BGR)
    cv2.imshow('Signs Detected', img_with_bbox_signs)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    if cv2.waitKey(0) & 0xFF == ord('s'):
        cv2.imwrite('detected_signs.jpg', img_with_bbox_signs)
        print("Saved image as detected_signs.jpg")