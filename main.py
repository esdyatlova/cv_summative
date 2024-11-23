import numpy as np
import cv2

img = cv2.imread("test_img_1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
h, w, _ = img.shape

contours_all, i = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

lines = []
letters = [np.array([])]

for i in range(h-1):
    #print(np.mean(img[i, :]), i)
    if np.mean(img[i, :]) < 253.0:
        if np.mean(img[i+1, :]) < 253.0:
            if len(lines[-1]) != 0:
                lines[-1] = np.hstack((lines[-1], img[i, :]))
            else:
                lines[-1] = img[i, :].copy()
    else:
        if np.mean(img[i + 1, :]) < 253.0:
            lines.append(np.array([]))

for i in range(len(lines)):
    #cv2.imshow("test" + str(i), cv2.rotate((cv2.flip(lines[i], 0)), 0))
    img_line = cv2.rotate((cv2.flip(lines[i], 0)), 0)
    h, w = img_line.shape
    for j in range(w-1):
        if np.mean(img_line[:, j]) < 241.0:
            if np.mean(img_line[:, j+1]) < 241.0:
                if len(letters[-1]) != 0:
                    letters[-1] = np.vstack([letters[-1], img_line[:, j]])
                else:
                    letters[-1] = img_line[:, j].copy()
        else:
            if np.mean(img_line[:, j+1]) < 241.0:
                letters.append(np.array([]))


for i in range(len(letters)):
    if len(letters[i] != 0):
        cv2.imshow("test" + str(i), cv2.rotate((cv2.flip(letters[i], 0)), 0))

cv2.imshow("original", img)

cv2.waitKey(0)