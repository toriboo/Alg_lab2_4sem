import file1_lab2 as func
from  PIL import Image
import matplotlib.pyplot as plt
'''matrix = [[1,2,3,4],[6,7,8,9],[11,12,13,14],[1,1,1,1]]
func.zigzag(matrix)'''
quant_matrix_CbCr = [[17,18,24,47,99,99,99,99],[18,21,26,66,99,99,99,99],[24,26,56,99,99,99,99,99],[47,66,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99]]
quant_matrix_Y = [[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]]
file_name = "Lenna.png"
im_arr = func.convert_image(file_name) #получение ЧБ, grayscale и т.д.
im_YCbCr= func.convert_to_ycbcr(im_arr[0]) # переход из RGB в YCbCr
Y, Cb, Cr = func.channel_ycbcr(im_YCbCr) # получение каналов Y, Cb, Cr
#print(Y)
Cb = func.downsempling(Cb,2) # даунсемплинг хроматических каналов
Cr = func.downsempling(Cr,2)
#im_Y, im_Cb, im_Cr = func.channel_image(Y, Image.fromarray(func.downsempling(Cb,2)),Image.fromarray(func.downsempling(Cr,2)), im_YCbCr)# олучения изорбажения с даунсемплингом
arr_blocks_Cb = func.get_blocks(Cb)# массив блоков 8*8 канала im_Cb
arr_blocks_Cr = func.get_blocks(Cr)# массив блоков 8*8 канала im_Cr
arr_blocks_Y = func.get_blocks(Y)# массив блоков 8*8 канала im_Y
#print(arr_blocks_Y[0])
#дискретно-косинусное преобразование для блоков 8*8
dct_Y = [func.DCT_2(x) for x in arr_blocks_Y]
dct_Cb = [func.DCT_2(x) for x in arr_blocks_Cb]
dct_Cr= [func.DCT_2(x) for x in arr_blocks_Cr]
# квантование коэффициентов DCT
C = 20 #- уровень качества
coeff_quant_matrix_Y = func.quant_coeff(quant_matrix_Y,C)
coeff_quant_matrix_CbCr = func.quant_coeff(quant_matrix_CbCr,C)
quant_Y = [func.quantization(x, coeff_quant_matrix_Y) for x in dct_Y]
quant_Cb = [func.quantization(x,coeff_quant_matrix_CbCr) for x in dct_Cb]
quant_Cr = [func.quantization(x,coeff_quant_matrix_CbCr) for x in dct_Cr]
print(quant_Y[0])
'''plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
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