import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ok, first_frame = cap.read()
frame_gray_init = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(first_frame)
#print(np.shape(first_frame))
#print(np.shape(hsv))
#print(first_frame)
#print(hsv)
hsv[...,1] = 255
#print(hsv)

def compute_dense_optical_flow(prev_image, current_image):
  old_shape = current_image.shape
  prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
  current_image_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
  assert current_image.shape == old_shape
  hsv = np.zeros_like(prev_image)
  hsv[..., 1] = 255
  flow = None
  flow = cv2.calcOpticalFlowFarneback(prev=prev_image_gray,
                                      next=current_image_gray, flow=flow,
                                      pyr_scale=0.8, levels=15, winsize=5,
                                      iterations=10, poly_n=5, poly_sigma=0,
                                      flags=10)

  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  hsv[..., 0] = ang * 180 / np.pi / 2
  hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # https://docs.opencv.org/3.4.15/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
    # 30 -> 10 -> 3
    #flow = cv2.calcOpticalFlowFarneback(frame_gray_init, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    flow =compute_dense_optical_flow(frame_gray_init, frame_gray)

    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1]) # X, Y
    hsv[...,0] = angle * (180 / (np.pi / 2))
    hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    frame_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Dense optical flow', frame_rgb)
    if cv2.waitKey(1) == 13: # enter
        break

    frame_gray_init = frame_gray

cap.release()
cv2.destroyAllWindows()



















