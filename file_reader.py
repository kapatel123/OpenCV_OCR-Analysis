import cv2
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes


class FileInputProcessor:
    @staticmethod
    def pdf2image_converter(pdf_fn):
        if pdf_fn is not None:
            return convert_from_path(pdf_fn)
        else:
            raise ValueError("Invalid data provided. Must have path to file")

    @staticmethod
    def img_converter(img_fn):
        if img_fn is not None:
            return cv2.imread(img_fn)
        else:
            raise ValueError("Invalid data provided. Must have path to file")
