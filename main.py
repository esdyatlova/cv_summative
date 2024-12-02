import numpy as np
import cv2

img = cv2.imread("all_capital_letters.jpg") #чтение файла (изображения)

def split(img): #функция делит текст на символы
    lines = [np.array([])]
    letters = [np.array([])]
    h, w, _ = img.shape

    for i in range(h-1):
        #print(np.mean(img[i, :]), i)
        if np.mean(img[i, :]) < 254.0:
            if np.mean(img[i+1, :]) < 254.0:
                if len(lines[-1]) != 0:
                    lines[-1] = np.hstack((lines[-1], img[i, :]))
                else:
                    lines[-1] = img[i, :].copy()
        else:
            if np.mean(img[i + 1, :]) < 254.0 and len(lines[-1]) != 0:
                lines.append(np.array([]))

    for i in range(len(lines)):
        #cv2.imshow("test" + str(i), cv2.rotate((cv2.flip(lines[i], 0)), 0))
        img_line = cv2.rotate((cv2.flip(lines[i], 0)), 0)
        h, w = img_line.shape
        for j in range(w-1):
            if np.mean(img_line[:, j]) < 236.5:
                if np.mean(img_line[:, j+1]) < 236.5:
                    if len(letters[-1]) != 0:
                        letters[-1] = np.vstack([letters[-1], img_line[:, j]])
                    else:
                        letters[-1] = img_line[:, j].copy()
            else:
                if np.mean(img_line[:, j+1]) < 236.5:
                    letters.append(np.array([]))
    return letters

def incnt(im): #подсчет количества внутренних контуров у буквы
    ret, thresh = cv2.threshold(im, 27, 255, cv2.THRESH_BINARY_INV)
    contours_all, i = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_ex, i2 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours_all) - len(contours_ex)

def vertical_lines(im): #подсчет количества вертикальных линий у буквы
    _, w = im.shape
    v_lines = [np.array([])]

    for j in range(w - 1):
        thr = 60.0
        if np.mean(im[:, j]) < thr:
            if np.mean(im[:, j + 1]) < thr:
                if len(v_lines[-1]) != 0:
                    v_lines[-1] = np.vstack([v_lines[-1], im[:, j]])
                else:
                    v_lines[-1] = im[:, j].copy()
        else:
            if np.mean(im[:, j + 1]) < thr:
                v_lines.append(np.array([]))
    if len(v_lines[-1]) != 0:
        return len(v_lines) - 1
    else:
        return 0

def horizontal_lines(im): #подсчет количества горизонтальных линий у буквы
    h, _ = im.shape
    h_lines = [np.array([])]
    for i in range(h-1):
        thr = 60.0
        if np.mean(im[i, :]) < thr:
            if np.mean(im[i+1, :]) < thr:
                if len(h_lines[-1]) != 0:
                    h_lines[-1] = np.hstack((h_lines[-1], im[i, :]))
                else:
                    h_lines[-1] = im[i, :].copy()
        else:
            if np.mean(im[i + 1, :]) < thr and len(h_lines[-1]) != 0:
                h_lines.append(np.array([]))
    if len(h_lines[-1]) != 0:
        return len(h_lines)
    else:
        return 0

def first_v_line(im): #есть ли у буквы слева черная линия (или что-то похожее на нее)
    h, w = im.shape

    thr = 150.0
    white = 236.0
    u = 0
    while np.mean(im[:, u]) > white:
        if u <= w-2:
            u += 1
        else:
            break
    if np.mean(im[:, u+2]) < thr:
        return 1
    else:
        return 0


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

letters = split(img)

for i in range(len(letters)):

    letter = cv2.rotate((cv2.flip(letters[i], 0)), 0)

    if len(letters[i]) != 0:
        #cv2.imshow("test" + str(i), letter)
        print(first_v_line(letter), i)

cv2.waitKey(0)