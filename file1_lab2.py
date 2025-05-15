import numpy as np
from PIL import Image
import numpy as np
import functools as ft


def convert_image(file_name):
    im_RGB = Image.open(file_name)
    im_GS = im_RGB.convert("L") # конвертация в grayscale
    im_D  = im_RGB.convert("1") # конвертация в ЧБ с встроенным дизерингом
    im_WD  = im_RGB.convert("1", dither = Image.Dither.NONE) # конвертация в grayscale и обработка каждого пикселя пороговой функцией -> переход в чб
    im_arr = [im_RGB, im_GS, im_D,im_WD]
    return im_arr
def RGB_to_YCbCr(R, G, B):
    R: float
    G: float
    B: float
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = - 0.168736 * R - 0.331264 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128
    return min(max(round(Y),0),255), min(max(round(Cb),0),255),min(max(round(Cr),0),255)
def YCbCr_to_RGB(Y,Cb,Cr):
    Y: float
    Cb: float
    Cr: float
    R = Y + 1.402*(Cr-128)
    G = Y - 0.714114*(Cr-128)- 0.034414*(Cb-128)
    B = Y + 1.772*(Cb-128)
    return min(max(round(R),0),255), min(max(round(G),0),255),min(max(round(B),0),255)
def convert_to_ycbcr(im_RGB):
    arr_pix = im_RGB.load()
    width, height = im_RGB.size
    ycbcr_arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Проходим по каждому пикселю и преобразуем его в YCbCr
    for i in range(height):
        for j in range(width):

            R,G,B = arr_pix[i,j]
            Y,Cb,Cr = RGB_to_YCbCr(R,G,B)
            ycbcr_arr[j, i] = [Y, Cb, Cr]
    im_ycbcr = Image.fromarray(ycbcr_arr, mode='YCbCr')
    return im_ycbcr
def convert_to_RGB(Y,Cb,Cr):
    width, height = Y.shape
    Y = np.array(Y)
    Cb = np.array(Cb)
    Cr = np.array(Cr)
    RGB_arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Проходим по каждому пикселю и преобразуем его в YCbCr
    for i in range(height):
        for j in range(width):
            y_value = Y[i][j]
            cb_value = Cb[i][j]
            cr_value = Cr[i][j]
            R, G, B = YCbCr_to_RGB(y_value, cb_value, cr_value)
            RGB_arr[i, j] = [R, G, B]
    im_RGB = Image.fromarray(RGB_arr, mode='RGB')
    return im_RGB
def channel_ycbcr(im_YCbCr):
    Y, Cb, Cr = im_YCbCr.split()
    Y = np.array(Y)
    Cb = np.array(Cb)
    Cr = np.array(Cr)
    return Y, Cb, Cr
def downsempling(matrix, k):
    matrix = np.array(matrix)
    width, height = matrix.shape
    new_matrix = matrix
    for i in range(height):
        for j in range(width):
            if (i % k == 0):
                new_matrix[i,j] = matrix[i,j-j%k]
            else:
                new_matrix[i,j] = matrix[i-1,j]
    return new_matrix
def channel_image(Y,Cb,Cr, im_YCbCr):
    const_ch = Image.new("L",im_YCbCr.size, 128)
    im_Y = Image.merge("YCbCr", (Y, const_ch, const_ch)).convert("RGB")
    im_Cb = Image.merge("YCbCr", (const_ch, Cb, const_ch)).convert("RGB")
    im_Cr = Image.merge("YCbCr", (const_ch, const_ch, Cr)).convert("RGB")
    return im_Y, im_Cb, im_Cr
def get_blocks(Img):
    arr_blocks = []
    arr_im = np.array(Img)
    width, height = arr_im.shape

    while width % 8 !=0:
        arr_im = np.insert(arr_im, [width], [0], axis = 0)
        width+=1
    while  height % 8 !=0:
        arr_im = np.insert(arr_im, [height], [0],axis = 1)
        height +=1

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            arr_ = arr_im[i:i + 8,j:j + 8]
            arr_blocks.append((arr_))
    return arr_blocks
def merge_blocks(arr, width, height):

    # Создаем пустую матрицу нужного размера
    image = np.zeros((height, width), dtype=arr[0].dtype)

    block_index = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if block_index < len(arr):
                image[i:i + 8, j:j + 8] = arr[block_index]
                block_index += 1
            else:
                print(f"Warning: Not enough blocks in 'arr' to fill the image at position ({i}, {j})")
                return image[:height, :width]  # Возвращаем текущее состояние изображения

    return image[:height, :width]
def DCT_2(matrix):
    # Проверяем размерность входного массива
    mat = np.tile(np.arange(8), (8, 1))
    cos_jm = np.cos(np.pi * mat * (2 * mat.T + 1) / 16)
    cos_in = cos_jm.T

    alpha = np.ones((8, 8)) * np.sqrt(2 / 8)
    alpha[:, 0] = alpha[0, :] / np.sqrt(2)

    cos_jm_scaled = np.multiply(cos_jm, alpha)
    cos_in_scaled = cos_jm_scaled.T

    # Вычисление DCT
    dct_coeff = ft.reduce(np.dot, [cos_in_scaled, matrix, cos_jm_scaled])

    # Округляем коэффициенты
    dct_coeff = np.round(dct_coeff)
    return dct_coeff

def i_DCT_2(dct_coeff):
    mat = np.tile(np.arange(8), (8, 1))
    cos_jm = np.cos(np.pi * mat * (2 * mat.T + 1) / 16)
    cos_in = cos_jm.T

    alpha = np.ones((8, 8)) * np.sqrt(2 / 8)
    alpha[:, 0] = alpha[0, :] / np.sqrt(2)

    cos_jm_scaled = np.multiply(cos_jm, alpha)
    cos_in_scaled = cos_jm_scaled.T
    # Вычисление обратного DCT
    idct_coeff = ft.reduce(np.dot, [cos_in_scaled.T, dct_coeff, cos_jm_scaled.T])
    idct_coeff = np.round(idct_coeff)
    return idct_coeff
def quant_coeff(quant_matrix, C):
    if C==0:
        s = 5000
    elif C<50:
        s = 5000/C
    else:
        s = 200-2*C
    if C == 100:
        return np.array(quant_matrix)
    quant_matrix = np.array(quant_matrix)
    half_matrix = 50*np.ones_like(quant_matrix)
    return np.round(((s*quant_matrix+half_matrix)/100)).astype(int)
def quantization(matrix, quant_matrix):
    quant_matrix = np.array(quant_matrix)
    return np.round(matrix/quant_matrix)
def i_quantization(matrix, quant_matrix):
    quant_matrix = np.array(quant_matrix)
    matrix = np.array(matrix)
    return np.round(matrix*quant_matrix)
def zigzag(matrix):
    matrix = np.array(matrix)
    arr=[]
    width, height = matrix.shape
    for i in range(height):
        diag = [matrix[x,i-x] for x in range(i,-1,-1)]
        #print(diag)
        if (len(diag)%2==0): diag.reverse()
        arr+= diag

    for i in range (1,height):
        diag = [matrix[x,height-x+i-1]for x in range (height-1,i-1,-1)]
        #print(diag)
        if (len(diag)%2==0):diag.reverse()
        arr+= diag
    #print(arr)
    return arr
def decode_zigzag(arr, width, height):
    # Создаем пустую матрицу нужного размера
    matrix = np.zeros((height, width), dtype=arr.dtype)

    index = 0  # Индекс для обхода массива arr
    for i in range(height):
        diag = []
        # Заполняем диагональ сверху вниз
        for x in range(i + 1):
            if x < width and (i - x) < height:
                diag.append((x, i - x))

        if len(diag) % 2 == 0:
            diag.reverse()

        for x, y in diag:
            if index < len(arr):
                matrix[y][x] = arr[index]
                index += 1

    for i in range(1, width):
        diag = []
        # Заполняем диагональ снизу вверх
        for x in range(height - 1, i - 1, -1):
            if (x >= 0) and (i + height - 1 - x < width):
                diag.append((height - 1 - x, i + height - 1 - x))

        if len(diag) % 2 == 0:
            diag.reverse()

        for x, y in diag:
            if index < len(arr):
                matrix[y][x] = arr[index]
                index += 1
    return matrix
def difference_encoding(arr):
    arr = np.array(arr)
    new_arr = arr.astype(np.int16)
    new_arr[1:] = new_arr[1:] - new_arr[:-1]
    return new_arr
def difference_decoding(arr):
    arr= np.array(arr)
    new_arr = arr.copy()
    new_arr[0] = arr[0]
    for i in range(1,len(arr)):
        new_arr[i] = new_arr[i]+ new_arr[i-1]
    return new_arr
def variable_encoding(x):
    x = np.array(x).astype(int)
    results = []
    for value in x:
        if value == 0:
            l = 0
        else:
            l = 0
            while np.abs(value) >= (2 ** l) :
                l += 1

        if value < 0:
            value = (value + (2 ** (l)))-1
        if l == 0:
            bit_code = '0'
        else:
            bit_code = f'{value:0{l}b}'  # Форматируем в двоичное представление с нужным количеством битов

        results.append((l, bit_code))

    return results
def variable_decoding(x):
    results = []
    for l, bit_code in x:
        if bit_code != '':
            value = int(bit_code,2)
            l = int(l)
            if l == 0:
                value = 0
            elif value < 2 ** (l-1):
                value = value - 2 ** l+1
        results.append(value)
    return results
def rle(arr):
    arr = np.array(arr)
    res = []
    counter = 0
    for coeff, bit_code in arr:
        coeff = int(coeff)
        if coeff == 0:
            counter += 1
            if counter == 16:
                res.append(('F/0',str(bit_code)))
                counter = 0
        else:
            if counter > 0:
                res.append((f'{hex(counter)[2:].upper()}/{hex(coeff)[2:].upper()}',str(bit_code)))
            else:
                res.append((f'0/{hex(coeff)[2:].upper()}',str(bit_code)))
            counter = 0
    if counter > 0:
        if counter == 16:
            res.append(('F/0',str(bit_code)))
        else:
            while counter > 0:
                (res.append((f'{hex(0)[2:].upper()}/0',str(bit_code))))
                counter -=1
    return res
def rle_decode(encoded):
    decoded = []
    for run_size, bite_code in encoded:
        run_length, value = run_size.split('/')
        run_length = int(run_length, 16)  # Преобразуем шестнадцатеричное значение в десятичное
        for i in range(run_length):
            decoded.append((0, 0))
        decoded.append((int(value,16),bite_code))
    return np.array(decoded)
def read_table_from_file(filename):
    table = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 3:
                    value0 = (parts[0])
                    value1 = (parts[1])
                    value2 = parts[2]
                    table[value0] = (value1, value2)
                    #print(table[value0])
    #print(table)
    return table
def Haffman(filename, arr):
    table = read_table_from_file(filename)
    encoded_result = []
    for l, bit_code in arr:
        str_l = str(l)
        if str_l in table:
            encoded_result.append(table[str_l][1])
            encoded_result.append(bit_code)

    # Объединяем все закодированные слова в одну строку
    coded_message = ''.join(encoded_result)
    coded_message += (8 - len(coded_message) % 8) * "0"
    bytes_string = b""
    for i in range(0, len(coded_message), 8):
        s = coded_message[i:i + 8]
        x = string_binary_to_int(s)
        bytes_string += x.to_bytes(1, "big")
    return bytes_string
def Haffman_AC(filename, arr):
    table = read_table_from_file(filename)
    encoded_result = []

    for l, bit_code in arr:
        if l in table:
            encoded_result.append(table[l][1])
            encoded_result.append(str(bit_code))
    # Объединяем все закодированные слова в одну строку
    coded_message = ''.join(encoded_result)
    coded_message += (8 - len(coded_message) % 8) * "0"
    bytes_string = b""
    for i in range(0, len(coded_message), 8):
        s = coded_message[i:i + 8]
        x = string_binary_to_int(s)
        bytes_string += x.to_bytes(1, "big")
    return bytes_string
def string_binary_to_int(s):
        X = 0
        for i in range(8):
            if s[i] == "1":
                X = X + 2**(7-i)
        return X
def bytes_to_binary(n):
    # Преобразуем байты в двоичную строку
    return ''.join(format(byte, '08b') for byte in n)
def count_symb(S):
    N = len(S)
    counter = np.array([0 for _ in range(256)])
    for s in S:
        counter[(s)] += 1
    return counter
def HA_decoding(compressed_text, codes):
    reverse_codes = {x[1]: y for y,x in codes.items()}
    res = []
    text = bytes_to_binary(compressed_text)
    current = ''
    i = 0
    while (i < len(text)):
        current += text[i]
        if (current in reverse_codes.keys()):
            c  = int(reverse_codes[current])
            current = ''
            if c == 0:
                bit_code = text[i + 1:i + 2]
                i+=1
            else:
                bit_code = text[i+1:i+1+c]
            res.append((c, bit_code))
            i+=c
        i +=1
    return res
def HA_decoding_AC(compressed_text, codes):
    reverse_codes = {x[1]: y for y,x in codes.items()}
    res = []
    text = bytes_to_binary(compressed_text)
    current = ''
    i = 0
    while (i < len(text)):
        current += text[i]
        if (current in reverse_codes.keys()):
            run_size = (reverse_codes[current])
            current = ''
            run,c = run_size.split("/")
            c = int(c,16)
            if c == 0:
                bit_code = text[i + 1:i + 2]
                i+=1
            else:
                bit_code = text[i+1:i+1+c]
                i+=c
            res.append((run_size,bit_code))
        i +=1
    return res

