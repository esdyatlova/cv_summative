import numpy as np
import cv2

img = cv2.imread("test5.jpg", cv2.IMREAD_GRAYSCALE) #чтение файла (изображения)


letters_dict = {"0111111":"|",
                "1010000":"А",
                "1131110":"Б",
                "2131110":"В",
                "0111100":"Г",
                "1020100":"Д",
                "0131110":"Е",
                "0100000":"Ж",
                "0000111":"З",
                "0201001":"И",
                "0101100":"К",
                "0110101":"Л",
                "0211001":"Н",
                "1001111":"О",
                "0211101":"П",
                "1121100":"Р",
                "0001110":"С",
                "0110100":"Т",
                "0000000":"У",
                "2121000":"Ф",
                "0011000":"Ц",
                "0111001":"Ч",
                "0311011":"Ш",
                "1100010":"Ъ",
                "1121010":"Ь",
                "0000110":"Э",
                "1101111":"Ю",
                "1121101":"Я"}

def split(im): #функция делит текст на символы
    #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    letters = []

    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])

        if hierarchy[0][i][3] == 0:
            letters.append((x, y, np.array(im[y:y+h, x:x+w])))

    letters.sort(key=lambda elem: (round(elem[1]/100), elem[0]), reverse=False)

    return letters


def internal_contours(im): #подсчет количества внутренних контуров у буквы
    ret, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
    contours_all, i = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_ex, i2 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return str(len(contours_all) - len(contours_ex))


def vertical_lines(im): #подсчет количества вертикальных линий у буквы
    _, w = im.shape
    v_lines = [np.array([])]

    for j in range(w - 1):
        thr = 40.0
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
        return str(len(v_lines) - 1)
    else:
        return "0"


def horizontal_lines(im): #подсчет количества горизонтальных линий у буквы
    h, _ = im.shape
    h_lines = [np.array([])]
    for i in range(h-1):
        thr = 110.0
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
        return str(len(h_lines))
    else:
        return "0"


def first_v_line(im): #есть ли у буквы слева черная линия (или что-то похожее на нее)
    _, w = im.shape

    thr = 150.0
    white = 236.0
    u = 0
    while np.mean(im[:, u]) > white:
        if u <= w-2:
            u += 1
        else:
            break
    if np.mean(im[:, u+2]) < thr:
        return "1"
    else:
        return "0"


def first_h_line(im): #есть ли у буквы сверху черная линия (или что-то похожее на нее)
    h, _ = im.shape

    thr = 150.0
    white = 236.0
    u = 0
    while np.mean(im[u, :]) > white:
        if u <= h - 2:
            u += 1
        else:
            break
    if np.mean(im[u + 2, :]) < thr:
        return "1"
    else:
        return "0"


def last_h_line(im): #есть ли у буквы справа черная линия (или что-то похожее на нее)
    h, _ = im.shape

    thr = 150.0
    white = 236.0
    u = 0
    while np.mean(im[h-u-1, :]) > white:
        if u < h-1:
            u += 1
        else:
            break
    if np.mean(im[h - u - 2, :]) < thr:
        return "1"
    else:
        return "0"


def last_v_line(im): #есть ли у буквы справа черная линия (или что-то похожее на нее)
    _, w = im.shape

    thr = 150.0
    white = 236.0
    u = 0
    while np.mean(im[:, w-u-1]) > white:
        if u < w - 1:
            u += 1
        else:
            break
    if np.mean(im[:, w - u - 2]) < thr:
        return "1"
    else:
        return "0"

def ts_tsh(im):
    _, w = im.shape
    v_lines = [np.array([])]

    for j in range(w - 1):
        thr = 70.0
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
        return str(len(v_lines) - 1)
    else:
        return "0"


def defining_letter(incnt, vnum, hnum, fvline, fhline, lhline, lvline, tstsh):
    key = incnt + vnum + hnum + fvline + fhline + lhline + lvline
    if letters_dict.get(key) is not None:
        ans = letters_dict.get(key)
    else:
        min = 8
        for item in letters_dict:
            dif = 0
            if item[0] == key[0]:
                for i in range(7): # 7 - количество символов в индивидуальных кодах букв
                    if item[i] != key[i]:
                        dif += 1
                if dif < min:
                    min = dif
                    min_item = item
                elif dif == min:
                    if abs(int(item) - int(key)) < abs(int(min_item) - int(key)):
                        min_item = item
        ans = letters_dict.get(str(min_item))
    if ans == "Ц":
        if tstsh == "3":
            ans = "Щ"
    return ans

letters = split(img)
text = ""

for i in range(len(letters)):

    letter = letters[i][2]

    if len(letters[i]) != 0 and np.shape(letter)[0] > 10:
        #cv2.imshow("test" + str(i), letter)
        text += defining_letter(internal_contours(letter), vertical_lines(letter),
                               horizontal_lines(letter), first_v_line(letter),
                               first_h_line(letter), last_h_line(letter), last_v_line(letter), ts_tsh(letter))
    text = text.replace("Ь|", "Ы")


print(text)

cv2.waitKey(0)