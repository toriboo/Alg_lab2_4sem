import file1_lab2 as func
from  PIL import Image
import numpy as np
import matplotlib.pyplot as plt

quant_matrix_CbCr = [[17,18,24,47,99,99,99,99],
                     [18,21,26,66,99,99,99,99],
                     [24,26,56,99,99,99,99,99],
                     [47,66,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99]]
quant_matrix_Y = [[16,11,10,16,24,40,51,61],
                  [12,12,14,19,26,58,60,55],
                  [14,13,16,24,40,57,69,56],
                  [14,17,22,29,51,87,80,62],
                  [18,22,37,56,68,109,103,77],
                  [24,35,55,64,81,104,113,92],
                  [49,64,78,87,103,121,120,101],
                  [72,92,95,98,112,100,103,99]]
file_name = "My.png"

im0 = Image.open("Lenna.png").convert("RGB")
im1 = Image.open("LennaGS.png").convert("RGB")
im2 = Image.open("LennaD.png").convert("RGB")
im3 = Image.open("LennaWD.png").convert("RGB")
def compress(im2,C):
    width,height  = im2.size
    im_YCbCr= func.convert_to_ycbcr(im2) # переход из RGB в YCbCr
    Y, Cb, Cr = func.channel_ycbcr(im_YCbCr) # получение каналов Y, Cb, Cr
    #print(im_YCbCr.size)
    #print(Y)
    Cb = func.downsempling(Cb,2) # даунсемплинг хроматических каналов
    Cr = func.downsempling(Cr,2)
    #im_Y, im_Cb, im_Cr = func.channel_image(Y, Image.fromarray(func.downsempling(Cb,2)),Image.fromarray(func.downsempling(Cr,2)), im_YCbCr)# олучения изорбажения с даунсемплингом
    arr_blocks_Cb = func.get_blocks(Cb)# массив блоков 8*8 канала im_Cb
    arr_blocks_Cr = func.get_blocks(Cr)# массив блоков 8*8 канала im_Cr
    arr_blocks_Y = func.get_blocks(Y)# массив блоков 8*8 канала im_Y
    #print(print("arr_blocks_Y =", len(arr_blocks_Y)))
    #print(arr_blocks_Y[1])

    #дискретно-косинусное преобразование для блоков 8*8
    dct_Y = [func.DCT_2(x) for x in arr_blocks_Y]
    dct_Cb = [func.DCT_2(x) for x in arr_blocks_Cb]
    dct_Cr= [func.DCT_2(x) for x in arr_blocks_Cr]
    #print("dct_Y =",len(dct_Y))
    #print(dct_Y[0],dct_Y[1])
    # квантование коэффициентов DCT

    C = C#- уровень качества
    coeff_quant_matrix_Y = func.quant_coeff(quant_matrix_Y,C)
    coeff_quant_matrix_CbCr = func.quant_coeff(quant_matrix_CbCr,C)
    quant_Y = [func.quantization(x, coeff_quant_matrix_Y) for x in dct_Y]
    quant_Cb = [func.quantization(x,coeff_quant_matrix_CbCr) for x in dct_Cb]
    quant_Cr = [func.quantization(x,coeff_quant_matrix_CbCr) for x in dct_Cr]
    #print("quant_Y =", len(quant_Y))
    #print(quant_Cr[0])
    #print(func.i_quantization(quant_Cr[0],coeff_quant_matrix_CbCr))
    #обход зигзагом
    zigzag_Y = [func.zigzag(x)for x in quant_Y]
    zigzag_Cb = [func.zigzag(x)for x in quant_Cb]
    zigzag_Cr = [func.zigzag(x)for x in quant_Cr]
    #print("zigzag_Y =", len(zigzag_Y))
    #print(zigzag_Y[0])
    #print(zigzag_Y)
    #Dc и Ас коэфф
    DC_Y = [x[0]for x in zigzag_Y]
    #print("DC_Y =",len(DC_Y))
    #print(DC_Y)
    AC_Y = [x[1:64] for x in zigzag_Y]
    AC_Y = [i for x in AC_Y for i in x]
    DC_Cb = [x[0]for x in zigzag_Cb]
    AC_Cb = [x[1:64] for x in zigzag_Cb]
    AC_Cb = [i for x in AC_Cb for i in x]
    DC_Cr = [x[0]for x in zigzag_Cr]
    AC_Cr = [x[1:64] for x in zigzag_Cr]
    AC_Cr = [i for x in AC_Cr for i in x]
    #print("AC_Y =", len(AC_Y))
    # разностное кодирование DC
    diff_DC_Y = func.difference_encoding(DC_Y)
    diff_DC_Cb = func.difference_encoding(DC_Cb)
    diff_DC_Cr= func.difference_encoding(DC_Cr)
    print("diff_DC_Y =", len(diff_DC_Y))
    #print(diff_DC_Y)
    # переменное кодирование DC
    var_DC_Y = func.variable_encoding(diff_DC_Y)
    var_DC_Cb = func.variable_encoding(diff_DC_Cb)
    var_DC_Cr = func.variable_encoding(diff_DC_Cr)
    #print(diff_DC_Y[0:8])
    #print(var_DC_Y)
    print("var_DC_Y =", len(var_DC_Y))
    #print(func.variable_decoding(var_DC_Y[0:8]))
    #переменное кодирование АС
    var_AC_Y = func.variable_encoding((AC_Y))
    var_AC_Cb = func.variable_encoding(AC_Cb)
    var_AC_Cr = func.variable_encoding(AC_Cr)
    #print(var_AC_Y[0:8])
    #var_AC_Y = [x[0]for x in var_AC_Y]
    #var_AC_Cb = [x[0]for x in var_AC_Cb]
    #var_AC_Cr = [x[0]for x in var_AC_Cr]
    print("var_AC_Y =", len(var_AC_Y))
    #RlE - AC - коэффициентов
    RLE_AC_Y = func.rle((var_AC_Y))
    RLE_AC_Cb = func.rle((var_AC_Cb))
    RLE_AC_Cr = func.rle((var_AC_Cr))
    print("RLE_AC_Y =", len(RLE_AC_Y))
    #print("RLE_AC_Cb = ", len(RLE_AC_Cb))
    #print("RLE_AC_Cr = ", len(RLE_AC_Cr))
    #print(RLE_AC_Y)
    #Хаффман для DC коэффициентов
    filename = 'Haff_DC_Y.txt'
    table_DC_Y = func.read_table_from_file(filename)
    Haffman_Y_DC = func.Haffman(filename, var_DC_Y)
    filename = 'Haff_DC_CbCr.txt'
    table_DC_CbCr = func.read_table_from_file(filename)
    Haffman_Cb_DC = func.Haffman(filename, var_DC_Cb)
    Haffman_Cr_DC = func.Haffman(filename, var_DC_Cr)
    #Хаффман для  AC коэффициентов
    #print(Haffman_Y_DC)
    filename = 'Haff_AC_Y.txt'
    table_AC_Y = func.read_table_from_file(filename)
    Haffman_Y_AC = func.Haffman_AC(filename, RLE_AC_Y)
    filename = 'Haff_AC_CbCr.txt'
    table_AC_CbCr = func.read_table_from_file(filename)
    Haffman_Cb_AC = func.Haffman_AC(filename, RLE_AC_Cb)
    Haffman_Cr_AC = func.Haffman_AC(filename, RLE_AC_Cr)
    #print("Haff_Y_DC=", len(Haffman_Y_DC))
    #print("Haff_Y_AC =", len(Haffman_Y_AC))
    ##print(Haffman_Y_AC)
    #запись в файл
    import pickle

    def write_to_file(filename,width,height,coeff_quant_matrix_Y,coeff_quant_matrix_CbCr, huffman_codes_Y_DC, huffman_codes_CbCr_DC,
                      huffman_codes_Y_AC, huffman_codes_CbCr_AC,
                      Haffman_Y_DC,Haffman_Cb_DC, Haffman_Cr_DC,
                      Haffman_Y_AC, Haffman_Cb_AC, Haffman_Cr_AC):
        with open(filename, 'wb') as f:
            pickle.dump(width, f)
            pickle.dump(height, f)
            # Записываем таблицы квантования
            pickle.dump(coeff_quant_matrix_Y, f)
            pickle.dump(coeff_quant_matrix_CbCr, f)

            # Записываем таблицы кодов Хаффмана
            pickle.dump(huffman_codes_Y_DC, f)
            pickle.dump(huffman_codes_CbCr_DC, f)
            pickle.dump(huffman_codes_Y_AC,f)
            pickle.dump(huffman_codes_CbCr_AC, f)

            # Записываем закодированные данные
            pickle.dump((Haffman_Y_DC, Haffman_Cb_DC, Haffman_Cr_DC), f)
            pickle.dump((Haffman_Y_AC, Haffman_Cb_AC, Haffman_Cr_AC), f)

    fn = str(C)+'_Lenna.txt'
    # Пример использования функции
    write_to_file(fn,width,height,coeff_quant_matrix_Y, coeff_quant_matrix_CbCr,table_DC_Y,table_DC_CbCr,table_AC_Y,table_AC_CbCr,Haffman_Y_DC,Haffman_Cb_DC, Haffman_Cr_DC,
                      Haffman_Y_AC, Haffman_Cb_AC, Haffman_Cr_AC)
    m = open(fn,'rb')
    e = m.read()
    print((C, len(e)))
for i in range(0,110,20):
    compress(im0, i)


'''plt.figure(figsize=(10,5))
plt.subplot(1,2,1)`
plt.title('Оригинальное изображение')
plt.imshow(im_Cr)
plt.axis('off')
plt.subplot(1,2,2)
plt.title('Изображение в YCbCr')

plt.imshow(im_Cb)
plt.axis('off')
plt.show()

print(np.array(arr_blocks[0]))
m= DCT_2(np.array(arr_blocks[0]))
##print(m)
i_m= i_DCT_2(m)
print(i_m)'''