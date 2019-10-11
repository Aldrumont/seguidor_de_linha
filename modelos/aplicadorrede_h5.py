from keras.models import load_model
import cv2
import numpy as np
from configparser import ConfigParser
import argparse
import sys
import logging

def main():
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument('--img_path', help='caminho imagem')
    argumentos.add_argument('--invert_color', type=bool, default=False,
                        help='inverter cor?')
    args = argumentos.parse_args()
    sys.stdout.write(str(predict_img(args)))


def predict_img(args):
    aux = []
    img = cv2.imread('fotos/parada/frame_450_0.213125.jpg', 0)
    if args.invert_color:
        img = (255-img)
    aux.append(img)
    img = np.array(aux)
    img = img/255
    img = img.reshape(img.shape[0], img.shape[1], img.shape[2], 1).astype('float32')
    pred = model.predict(img)
    # pred = np.around(pred, decimals = 2, out = None) 
    tipos = ['cruzamento', 'curva', 'parada', 'reta', 'zigzag']
    print(tipos[np.argmax(pred)], pred)
    return (tipos[np.argmax(pred)], pred)


if __name__ == '__main__':
    logging.basicConfig(filename="test.log", level=logging.INFO, format='%(asctime)s - %(message)s')
    # logger = logging.basicConfig()

    parser = ConfigParser()
    parser.read('config.ini')
    model = load_model(parser.get("modelos","modelo_h5"))

    logging.debug('LEU TUDO')
    main()
