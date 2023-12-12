import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node')
        self.publisher_ = self.create_publisher(Image, 'processed_image', 10)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.image_path = "/home/eliezer/Documentos/IA/imagen2.jpg"
        self.bridge = CvBridge()

    def timer_callback(self):
        #Lectura de la imagen
        img = cv2.imread(self.image_path)

        if img is None:
            self.get_logger().info(f'Error cargando la imagen desde {self.image_path}')
            rclpy.shutdown()
            return

        # Cortar la imagen
        img_cortada = self.cortar_imagen(img)

        # Aplicación de la homografía
        img_hom = self.homografia_imagen(img_cortada)

        # Detectar y trazar líneas
        resultado = self.detectar_y_trazar_lineas(img_hom)

        # Se publica la imagen ya procesda
        img_msg = self.bridge.cv2_to_imgmsg(resultado, "bgr8")
        self.publisher_.publish(img_msg)
        self.get_logger().info('Publicando una imagen procesadaaaaaaaa')

    def cortar_imagen(self, imagen):
        h, w, _ = np.shape(imagen)
        imgcopy = imagen[int(4 * h / 8):h, int(3 * w / 8):w, :].copy()
        return imgcopy

    def homografia_imagen(self, imagen):
        scale = 4
        img_rs = cv2.resize(imagen, None, fx=1. / scale, fy=1. / scale, interpolation=cv2.INTER_LANCZOS4)
        b, g, r = cv2.split(img_rs)
        img_rs = cv2.merge([r, g, b])
        rows = img_rs.shape[0]
        cols = img_rs.shape[1]
        pts1 = np.float32([[0, 160], [0, rows], [cols, 160], [cols, rows]])
        x = 191
        pts2 = np.float32([[0, 0], [x, rows], [cols, 0], [cols - x, rows]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_hom = cv2.warpPerspective(img_rs, M, (cols, rows))
        return img_hom

    def detectar_y_trazar_lineas(self, imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(desenfoque, 150, 350)
        lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=50)
        imagen_con_lineas = imagen.copy()

        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            angulo_rad = np.arctan2(y2 - y1, x2 - x1)
            angulo_grados = np.degrees(angulo_rad) * (-1)
            if angulo_grados != 0:
                cv2.line(imagen_con_lineas, (x1, y1), (x2, y2), (0, 0, 255), 2)
                print(f'Angulo para la línea ({x1}, {y1}) - ({x2}, {y2}): {angulo_grados:.2f} grados')

        return imagen_con_lineas

def main(args=None):
    rclpy.init(args=args)
    image_processor_node = ImageProcessorNode()
    rclpy.spin(image_processor_node)
    image_processor_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
