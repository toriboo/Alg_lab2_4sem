import pickle
import file1_lab2 as func
import numpy as np
from  PIL import Image
import matplotlib.pyplot as plt
def read_from_file(filename):
    with open(filename, 'rb') as f:
        # Читаем таблицы квантования
        width= pickle.load(f)
        height = pickle.load(f)
        quant_matrix_Y = pickle.load(f)
        quant_matrix_CbCr = pickle.load(f)

        # Читаем таблицы кодов Хаффмана
        huffman_codes_Y_DC = pickle.load(f)
        huffman_codes_CbCr_DC = pickle.load(f)
        huffman_codes_Y_AC = pickle.load(f)
        huffman_codes_CbCr_AC = pickle.load(f)

        # Читаем закодированные данные
        Haffman_Y_DC, Haffman_Cb_DC, Haffman_Cr_DC = pickle.load(f)
        Haffman_Y_AC, Haffman_Cb_AC, Haffman_Cr_AC = pickle.load(f)

    return (width,height,quant_matrix_Y, quant_matrix_CbCr,
            huffman_codes_Y_DC, huffman_codes_CbCr_DC,
            huffman_codes_Y_AC, huffman_codes_CbCr_AC,
            Haffman_Y_DC, Haffman_Cb_DC, Haffman_Cr_DC,
            Haffman_Y_AC, Haffman_Cb_AC, Haffman_Cr_AC)


# Пример использования

def decompress(filename):
    data = read_from_file(filename)

    # Распаковка данных
    (width,height,quant_matrix_Y, quant_matrix_CbCr,
     huffman_codes_Y_DC, huffman_codes_CbCr_DC,
     huffman_codes_Y_AC, huffman_codes_CbCr_AC,
     Haffman_Y_DC, Haffman_Cb_DC, Haffman_Cr_DC,
     Haffman_Y_AC, Haffman_Cb_AC, Haffman_Cr_AC) = data
    #print(Haffman_Y_DC)
    #print("Haffman_Y_DC",len(Haffman_Y_DC))
    #print("Haffman_Y_AC",len(Haffman_Y_AC))
    #ltrjlХаффман для DC коэффициентов

    #print(Haffman_Cb_DC)
    var_DC_Y = np.array(func.HA_decoding(Haffman_Y_DC,huffman_codes_Y_DC))
    var_DC_Cb = np.array(func.HA_decoding(Haffman_Cb_DC,huffman_codes_CbCr_DC))
    var_DC_Cr = np.array(func.HA_decoding(Haffman_Cr_DC,huffman_codes_CbCr_DC))
    #print("var_Y_DC",len(var_DC_Y))
    #print(var_DC_Y)
    #декодирование AC хаффман
    RLE_AC_Y = func.HA_decoding_AC(Haffman_Y_AC,huffman_codes_Y_AC)
    RLE_AC_Cb = func.HA_decoding_AC(Haffman_Cb_AC,huffman_codes_CbCr_AC)
    RLE_AC_Cr = func.HA_decoding_AC(Haffman_Cr_AC,huffman_codes_CbCr_AC)
    #print(RLE_AC_Y)
    #декодирование RLE
    #print("RLE_AC_Y",len(RLE_AC_Y))
    #print("RLE_AC_Cb = ", len(RLE_AC_Cb))
    #print("RLE_AC_Cr = ", len(RLE_AC_Cr))
    #RlE - AC - коэффициентов

    var_AC_Y = func.rle_decode(RLE_AC_Y)
    #print(var_AC_Y[0:8])
    var_AC_Cb = func.rle_decode(RLE_AC_Cb)
    var_AC_Cr = func.rle_decode(RLE_AC_Cr)
    #print("var_AC_Y",len(var_AC_Y))
    diff_DC_Y = func.variable_decoding(var_DC_Y)
    diff_DC_Cb = func.variable_decoding(var_DC_Cb)
    diff_DC_Cr = func.variable_decoding(var_DC_Cr)
    #print(diff_DC_Y)
    #print("diff_Y_DC",len(diff_DC_Y))
    # РАЗБИТЬ НА МАССИВЫ ПО 64
    AC_Y = func.variable_decoding(var_AC_Y)
    AC_Cb = func.variable_decoding(var_AC_Cb)
    AC_Cr = func.variable_decoding(var_AC_Cr)
    #print("AC_Y",len(AC_Y))

    # декодирование
    DC_Y = func.difference_decoding(diff_DC_Y)
    DC_Cb = func.difference_decoding(diff_DC_Cb)
    DC_Cr = func.difference_decoding(diff_DC_Cr)
    #print("DC_Y",len(DC_Y))
    #print(DC_Y)
    # AC,DC коэфф
    #print(len(var_AC_Y))
    width = 2048
    height = 2048
    def conc_ac_dc(AC,DC):
        arr = []
        combined = []
        k =width*height-width*height//64
        for i in range(0, k, 63):
            ac = AC[i:i + 63]
            dc = DC[(i // 63) % len(AC)]
            combined.append(dc)
            combined.append(ac)
            combined = np.concatenate([[dc], ac])
            arr.append(combined)
            combined = []
        return arr
    #print(len(AC_Y)+len(DC_Y))
    zigzag_Y = conc_ac_dc(AC_Y,DC_Y)
    #print((zigzag_Y)[0:8])
    #print(zigzag_Y[0])
    zigzag_Cb = conc_ac_dc(AC_Cb,DC_Cb)
    zigzag_Cr = conc_ac_dc(AC_Cr,DC_Cr)
    #print("zigzag_Y",len(zigzag_Y))
    # декодирование зигзага
    quant_Y = [func.decode_zigzag(x,8,8) for x in zigzag_Y]
    quant_Cb = [func.decode_zigzag(x,8,8) for x in zigzag_Cb]
    quant_Cr = [func.decode_zigzag(x,8,8) for x in zigzag_Cr]
    #print(quant_Y[0:8])
    #print(quant_Y[0])
    #print("quant_Y",len(quant_Y))
    #декодирование матрицы квантования
    dct_Y = [func.i_quantization(x, quant_matrix_Y) for x in quant_Y]
    dct_Cb = [func.i_quantization(x,quant_matrix_CbCr) for x in quant_Cb]
    dct_Cr = [func.i_quantization(x,quant_matrix_CbCr) for x in quant_Cr]
    #print("dct_Y",len(dct_Y))
    #print(dct_Y)
    #print(len(dct_Y))
    #print(dct_Cb[0])
    #декодирование DCT
    arr_blocks_Y= [func.i_DCT_2(x) for x in dct_Y]
    arr_blocks_Cb= [func.i_DCT_2(x) for x in dct_Cb]
    arr_blocks_Cr= [func.i_DCT_2(x) for x in dct_Cr]
    #print("arr_blocks_Y",len(arr_blocks_Y))
    #arr_blocks_Y = (dct_Y[0],dct_Y[1])
    #print(arr_blocks_Y)
    #print(quant_Cr[0])
    #print(func.i_quantization(quant_Cr[0],coeff_quant_matrix_CbCr
    #обьединение блоков
    #print(len(arr_blocks_Y))
    Cb = func.merge_blocks(arr_blocks_Cb,2048,2048)# массив блоков 8*8 канала im_Cb
    Cr = func.merge_blocks(arr_blocks_Cr,2048,2048)# массив блоков 8*8 канала im_Cr
    Y = func.merge_blocks(arr_blocks_Y,2048,2048)# массив блоков 8*8 канала im_Y
    #print(Y[0:8][0:8])
    #print(Y.shape)
    im_RGB = func.convert_to_RGB(Y,Cb,Cr)
    im_RGB.save("Lenna_decode_"+filename[0:1]+".png")
    print("save ", filename)

for i in range(0,110,20):
    decompress(str(i)+'_Lenna.txt')