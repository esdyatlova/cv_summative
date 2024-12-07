#импорт библиотек numpy и cv2, необходимых для работы с изображениями и массивами, в которых хранятся изображения
import numpy as np
import cv2

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE) #чтение файла (изображения)
#для использования ВМЕСТО input.jpg впишите имя вашего файла!

# создание словаря где каждой букве соответствует индивидуальная последовательность цифр:
# 1. количество внутренних контуров
# 2. количество вертикальных линий
# 3. количество горизонтальных линий
# 4. самый левый контур буквы вертикальная линия или нет (1-да, 0-нет)
# 5. самый верхний контур буквы горизонтальная линия или нет (1-да, 0-нет)
# 6. самый нижний контур буквы горизонтальная линия или нет (1-да, 0-нет)
# 7. самый правый контур буквы вертикальная линия или нет (1-да, 0-нет)

letters_dict = {"0111111":"|", # буква Ы состоит из двух частей - Ь + |
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
    ret, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    letters = []
    w_mean = 0 #средняя ширина букв
    y_previous = 0 #y-координата предыдущей буквы
    line = 0 # номер строки на которой находится буква

    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])

        if hierarchy[0][i][3] == 0: #определение контура буквы
            if len(letters) != 0 and h > 20:
                if (y_previous - y) > h: #определение строки, на которой буква распологается
                    line += 1
                y_previous = y
            letters.append((x, line, np.array(im[y:y + h, x:x + w]), w)) #добавление буквы в список букв [координата по х, номер строки, изображение буквы, ширина буквы]

            w_mean = (w_mean * i + w)/(i+1) #определение средней ширины букв в строке

    if len(letters) != 0: #сортировка массива букв по х координатам и строке, в которой буква расположена
        letters.sort(key=lambda elem: (-elem[1], elem[0]), reverse=False)

        x_previous = letters[0][0] #переменная хранящая значение x
        w_previous = letters[0][3] #переменная хранящая значение длины
        d_mean = w_mean * 0.3 #среднее расстояние между символами

        for i in range(1, len(letters)): #определение переносов строки и пробелов
            x = letters[i][0]
            y = letters[i][1]
            w = letters[i][3]
            d = x - (x_previous + w_previous)
            if x - x_previous < 0 and not isinstance(letters[i-1][2], str): #перенос строки
                letters.insert(i, (x_previous + w, y, "enter"))
            elif d > 0 and d > d_mean and not isinstance(letters[i-1][2], str): #пробел
                letters.insert(i, (x_previous + w, y, "space"))

            x_previous = x
            w_previous = w

    return letters


def internal_contours(im): #подсчет количества внутренних контуров у буквы
    ret, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
    contours_all, i = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #подсчет количества всех контуров
    contours_ex, i2 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #подсчет количества внешних контуров

    return str(len(contours_all) - len(contours_ex)) #количество внутренних контуров = количество всех контуров - количество внешних контуров


def vertical_lines(im): #подсчет количества вертикальных линий у буквы
    _, w = im.shape
    v_lines = [np.array([])]

    for j in range(w - 1): #для каждого столбца в массиве хранящем изображение найдем среднее значение пикселей и сравним его с пороговым
        thr = 40.0 #пороговое значение
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
    for i in range(h-1):  #для каждой строки в массиве хранящем изображение найдем среднее значение пикселей и сравним его с пороговым
        thr = 100.0 #пороговое значение
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

    thr = 150.0 #пороговое значение линии
    white = 236.0 #пороговое значение белого
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

    thr = 150.0 #пороговое значение линии
    white = 236.0 #пороговое значение белого
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

    thr = 150.0 #пороговое значение линии
    white = 236.0 #пороговое значение белого
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

    thr = 170.0 #пороговое значение линии
    white = 236.0 #пороговое значение белого
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

def ts_tsh(im): # Ц или Щ
    h, w = im.shape
    im = (im[0:10, (w // 2 - 5):(w // 2 + 5)]) # проверяем попадает ли буква в небольшую область на самом верху изображения
    thr = 160.0 #пороговое значение до него попадает, после не попадает
    if np.mean(im) < thr:
        return "Щ" #Щ попадает
    else:
        return "Ц" #Ц нет

def z_e(im): # З или Э
    h, _ = im.shape
    h_lines = [np.array([])]
    for i in range(h - 1): #подсчет количества горизонтальных контуров, но с меньшим пороговым значением
        thr = 120.0
        if np.mean(im[i, :]) < thr:
            if np.mean(im[i + 1, :]) < thr:
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

def l_p(im): # Л или П
    _, w = im.shape
    #определение наличия вертикальной палки слева (у П есть, у Л нет)
    thr = 70.0 #пороговое значение линии
    white = 250.0 #пороговое значение белого
    u = 0
    while np.mean(im[:, u]) > white:
        if u <= w - 2:
            u += 1
        else:
            break
    if np.mean(im[:, u + 2]) < thr:
        return 1
    else:
        return 0

def m_i(im): # М или И
    h, w = im.shape
    im = (im[(h-10):h, (w//2-5):(w//2+5)]) #проверяем попадает ли буква в центральный нижний сектор (М - да, И - нет)
    thr = 160.0 # пороговое значение попадания
    if np.mean(im) < thr:
        return "М" # М - попадает
    else:
        return "И" # И - не попадает

def u_h(im): #У или Х
    h, w = im.shape
    im = (im[(h - 10):h, (w - 10):w]) #проверяем попадает ли буква в правый нижний сектор (Х - да, У - нет)
    thr = 180.0 #пороговое значение попадания
    if np.mean(im) < thr:
        return "Х" # Х - попадает
    else:
        return "У" # У - не попадает

def defining_letter(incnt, vnum, hnum, fvline, fhline, lhline, lvline, tstsh, ze, lp, mi, uh): # функция распознавания, получает на вход значения всех функций определения контуров буквы, возвращает букву
    key = incnt + vnum + hnum + fvline + fhline + lhline + lvline # получение индивидуального кода буквы
    if letters_dict.get(key) is not None: # поиск буквы по коду в словаре
        ans = letters_dict.get(key)
    else: # если буквы с таким кодом нет в словаре ищем максимально близкую
        min = 8 # количество символов в индивидуальных кодах букв +1
        for item in letters_dict:
            dif = 0 # разница между двумя кодами
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
    if ans == "Ц": # различение Ц и Щ
        ans = tstsh
    if ans == "З" or ans == "Э": # различение З и Э
        if ze == 3:
            ans = "Э"
        else:
            ans = "З"
    if ans == "Л" or ans == "П": # различение Л и П
        if lp == 1:
            ans = "П"
        else:
            ans = "Л"
    if ans == "И": # различение И и М
        ans = mi
    if ans == "У": # различение У и Х
        ans = uh
    return ans

# основная часть программы
letters = split(img) # разделение текста на символы, вызов функции split
text = "" # создание пустой строки, куда в дальнейшем будут добавляться символы

for i in range(len(letters)):

    letter = letters[i][2]

    if isinstance(letter, str): # определение пробелов и переноса строки
        if letter == "space":
            text += " "
        if letter == "enter":
            text += "\n"

    elif len(letters[i]) != 0 and np.shape(letter)[0] > 20: # распознавание букв
        # вызов функции определения буквы, и передача в нее значений всех функций участвующих в распознании контуров
        # добавление к переменной text значения выданного функцией defining_letter
        text += defining_letter(internal_contours(letter), vertical_lines(letter),
                               horizontal_lines(letter), first_v_line(letter),
                               first_h_line(letter), last_h_line(letter), last_v_line(letter), ts_tsh(letter),
                                z_e(letter), l_p(letter), m_i(letter), u_h(letter))
    text = text.replace("Ь|", "Ы") # объединение Ь и | в Ы


print(text) # вывод получившегося текста на экран