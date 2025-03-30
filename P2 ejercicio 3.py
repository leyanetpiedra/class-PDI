import cv2
import numpy as np

# Cargar imagen en color y en escala de grises
img_color = cv2.imread(r"C:\Users\Leyanet Piedra\Desktop\pdi lab\PDI\P2\ana.jpg")  # Imagen en formato BGR (color)

# Obtener dimensiones de la imagen
height, width, channels = img_color.shape  # Extrae altura, ancho y número de canales de la imagen
print(f'Dimensiones de la imagen: {width}x{height} con {channels} canales')

# Preprocesamiento: Convertir la imagen a escala de grises
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)  # Asegura que la imagen está en escala de grises

# Ajustar brillo y contraste (ecualización de histograma)
img_equalized = cv2.equalizeHist(img_gray)  # Aplica ecualización de histograma para mejorar el contraste

# Segmentación: Aplicar umbralización
valor_umbral = 128  # Valor de umbral para binarización
_, img_bin = cv2.threshold(img_equalized, valor_umbral, 255, cv2.THRESH_BINARY_INV)  # Umbralización inversa

# Separar canales de la imagen RGB
b, g, r = cv2.split(img_color)  # Separa los canales azul, verde y rojo de la imagen

# Análisis de características: Obtener el área del objeto segmentado
area = np.count_nonzero(img_bin)  # Cuenta los píxeles blancos en la imagen binaria (área del objeto)
print(f'Área del objeto segmentado: {area} píxeles')

# Extraer el bounding box del objeto
x, y, w, h = cv2.boundingRect(img_bin)  # Obtiene las coordenadas del rectángulo que encierra el objeto

# Dibujar el bounding box sobre la imagen en color
cv2.rectangle(img_color, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Dibuja un rectángulo azul alrededor del objeto

# Mostrar imágenes
cv2.imshow('Imagen Original', img_color)  # Muestra la imagen original con el bounding box
cv2.imshow('Imagen en Escala de Grises', img_gray)  # Muestra la imagen en escala de grises
cv2.imshow('Imagen Ecualizada', img_equalized)  # Muestra la imagen con ecualización de histograma
cv2.imshow('Imagen Binaria', img_bin)  # Muestra la imagen binarizada tras la umbralización
cv2.waitKey(0)  # Espera a que el usuario presione una tecla
cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas

