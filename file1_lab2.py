import numpy as np
from PIL import Image
import numpy as np
import functools as ft
import queue

def convert_image(file_name):
    im_RGB = Image.open(file_name)
    im_GS = im_RGB.convert("L") # конвертация в grayscale
    im_D  = im_RGB.convert("1") # конвертация в ЧБ с встроенным дизерингом
    im_WD  = im_RGB.convert("1", dither = Image.Dither.NONE) # конвертация в grayscale и обработка каждого пикселя пороговой функцией -> переход в чб
    im_arr = [im_RGB, im_GS, im_D,im_WD]
    return im_arr
def RGB_to_YCbCr(R, G, B):
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = - 0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
    return Y, Cb, Cr
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
    width, height = Img.shape
    arr_im = np.array(Img)
    while width % 8 !=0:
        np.insert(arr_im, [width], [0]*height, axis = 0)
        width+=1
    while  height % 8 !=0:
        np.insert(arr_im, [height], [0]*width,axis = 1)
        height +=1

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            arr = arr_im[i:i + 8,j:j + 8]
            arr_blocks.append((arr))
    return arr_blocks
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
    dct_coeff = np.round(dct_coeff).astype(int)

    return dct_coeff
def i_DCT_2(dct_coeff):
    if dct_coeff.ndim == 3:  # Если это цветное изображение
        X_new_1 = np.zeros_like(dct_coeff)
        for i in range(dct_coeff.shape[2]):  # Проходим по каждому каналу
            X_new_1[:, :, i] = i_DCT_2(dct_coeff[:, :, i])  # Применяем обратный DCT к каждому каналу
        return X_new_1

    mat = np.tile(np.arange(8), (8, 1))
    cos_jm = np.cos(np.pi * mat * (2 * mat.T + 1) / 16)
    cos_in = cos_jm.T

    alpha = np.ones((8, 8)) * np.sqrt(2 / 8)
    alpha[:, 0] = alpha[0, :] / np.sqrt(2)

    cos_jm_scaled = np.multiply(cos_jm, alpha)
    cos_in_scaled = cos_jm_scaled.T

    # Вычисление обратного DCT
    idct_coeff = ft.reduce(np.dot, [cos_in_scaled.T, dct_coeff, cos_jm_scaled.T])
    idct_coeff = np.round(idct_coeff).astype(int)
    return idct_coeff
def quant_coeff(quant_matrix, C):
    if C<50:
        s = 5000/C
    else:
        s = 200-2*C
    quant_matrix = np.array(quant_matrix)
    return s*quant_matrix
def quantization(matrix, quant_matrix):
    quant_matrix = np.array(quant_matrix)
    return np.round(matrix/quant_matrix)
def i_quantization(matrix, quant_matrix):
    return matrix*quant_matrix
def zigzag(matrix):
    matrix = np.array(matrix)
    arr=[]
    flag_reverse = False
    width, height = matrix.shape
    for i in range(height):
        diag = [matrix[x,i-x] for x in range(i,-1,-1)]
        print(diag)
        if (len(diag)%2==0): diag.reverse()
        arr+= diag

    for i in range (1,height):
        diag = [matrix[x,height-x+i-1]for x in range (height-1,i-1,-1)]
        print(diag)
        if (len(diag)%2==0):diag.reverse()
        arr+= diag
    print(arr)
    return arr
def difference_encoding(arr):
    new_arr = arr.astype(np.int16)
    new_arr[1:] = new_arr[1:] - new_arr[:-1]
    return new_arr
def difference_decoding(arr):
    new_arr = arr.copy()
    for i in range(1,len(arr)):
        new_arr[i] += new_arr[i-1]
    arr[1:]+= new_arr[:-1]
    return arr
class Node():
    def __init__(self, symbol = None, counter = None, left_child = None, right_child =None, parent = None):
        self.symbol = symbol
        self.counter = counter
        self.left = left_child
        self.right = right_child
        self.parent = parent
    def __lt__(self, other):
        return self.counter < other.counter
def rle(text):
    n = len(text)
    compressed_text = bytearray(b'')
    counter= 1
    prev_symbol = text[0]
    buffer = bytearray(b'')
    for i in range(1, n):
        if prev_symbol == text[i]:
            if (len(buffer) > 0):
                while (len(buffer) > 127):
                    compressed_text.append(127+128)
                    compressed_text.extend(buffer[:127])
                    buffer = buffer[127:]
                if (len(buffer)> 0):
                    compressed_text.append(len(buffer)+128)
                    compressed_text.extend(buffer)
                    buffer = bytearray(b'')
            counter += 1

        else:
            if (counter > 1):
                while (counter > 127):
                    compressed_text.append(127)
                    compressed_text.append(prev_symbol)
                    counter -= 127
                if (counter > 0):
                    compressed_text.append(counter)
                    compressed_text.append(prev_symbol)
                counter = 1
            else:
                buffer.append(prev_symbol)
        prev_symbol = text[i]

    if (len(buffer) > 0):
        buffer.append(prev_symbol)
        while (len(buffer) > 127):
            compressed_text.append(127 + 128)
            compressed_text.extend(buffer[:127])
            buffer = buffer[127:]
        if (len(buffer) > 0):
            compressed_text.append(len(buffer) + 128)
            compressed_text.extend(buffer)
    else:
        while (counter > 127):
            compressed_text.append(127)
            compressed_text.append(prev_symbol)
            counter -= 127
        if (counter > 0):
            compressed_text.append(counter)
            compressed_text.append(prev_symbol)
    return compressed_text
def rle_decoding(text):
    n = len(text)
    i =0
    decompressed_text = bytearray(b'')
    while (i<n):
        if (text[i]< 128):
            for j in range(text[i]):
                if(i+1 < n): decompressed_text.append(text[i + 1])
            i +=2
        else:
            for j in range(text[i]-128):
                if (i + 1 < n): decompressed_text.append(text[i+1])
                i += 1
            i+=1
    return decompressed_text
def Haffman_alg(text):
    n = len(text)
    Counters_symb  = count_symb(text)
    leafs = []
    q = queue.PriorityQueue()
    for i in range(256):
        if Counters_symb[i] != 0:
            node = Node(symbol=(i), counter=Counters_symb[i])
            leafs.append(node)
            q.put(node)
    while (q.qsize() >= 2):
        left_child = q.get()
        right_child = q.get()
        parent = Node(counter = left_child.counter + right_child.counter)
        parent.left_child = left_child
        parent.right_child = right_child
        left_child.parent = parent
        right_child.parent = parent
        q.put(parent)
    codes = {}
    for leaf in leafs:
        node = leaf
        code = ""
        while node.parent != None:
            if node.parent.left_child == node:
                code = "0" + code
            else:
                code = "1" + code
            node = node.parent
        codes[leaf.symbol] = code
    coded_message = ""
    for s in text:
        coded_message += codes[s]
    ##k = 8 - len(coded_message) % 8
    coded_message += (8 - len(coded_message) % 8) * "0"
    bytes_string = b""
    for i in range(0, len(coded_message), 8):
        s = coded_message[i:i + 8]
        x = string_binary_to_int(s)
        bytes_string += x.to_bytes(1, "big")
    return bytes_string, codes, n
def HA_decoding(compressed_text, codes, n):
    reverse_codes = {x: y for y,x in codes.items()}
    res = bytearray(b'')
    text = bytes_to_binary(compressed_text)
    current = ''
    i = 0
    while (len(res)<n and i < len(text)):
        current += text[i]
        if (current in reverse_codes.keys()):
            res.append(reverse_codes[current])
            current = ''
        i +=1
    return res
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


