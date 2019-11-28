import cv2
import numpy as np
from nltk.corpus import brown
import pytesseract
from PIL import Image
import re
from pdf2image import convert_from_path, convert_from_bytes
import io


class TextPreProcessor:
    def __init__(self):
        self.english_vocab = set(word.lower() for word in brown.words())

    def pdf2image_converter_from_path(self, pdf_file):
        if pdf_file is not None:
            return convert_from_path(pdf_file)
        else:
            raise ValueError("Invalid data provided.  Must be a path to a file")

    # Preprocess PDF Images
    def img_to_gray(self, image):
        # Apply grayscale filter to image. Return orig. image if issue arises
        new_image = None
        try:
            new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            new_image = image
        return new_image

    def blur_img(self, grayed_image):
        blurred_image = cv2.blur(grayed_image, (7, 7))
        return cv2.Canny(blurred_image, 80, 200)

    def bw_binary_img(self, edges):
        ret, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        return contours

    def sharpen_img(self, image):
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, sharpen_kernel)

    def find_max_contours(self, contours):
        contours.sort(key=len)
        return contours[-1]

    def draw_box(self, contours_list):
        rect = cv2.minAreaRect(contours_list)
        return rect[-1]

    def auto_angle_detection(self, img):
        angle_detect = []
        try:
            # convert return value to string
            angle_detect = str(pytesseract.image_to_osd(img))
        except:
            message = "Orientation not detected by PyTesseract, defaulting to manual angle detection"
        else:
            angle_detect = re.search(
                r"(?<=Orientation in degrees: )\d+", angle_detect
            ).group(0)
        return angle_detect

    def determine_orientation_angle(self, img, angle):
        (h, w) = img.shape[:2]
        angle_detect = self.auto_angle_detection(img)
        new_angle = int()

        # If PyTesseract is able to determine the orientation
        if type(angle_detect) == str and angle < 0:
            if int(angle_detect) == 0 and h > w:
                new_angle = 0.0
            elif int(angle_detect) == 90 and h > w:
                new_angle = 90.0
            elif int(angle_detect) == 180 and h > w:
                new_angle = -180.0
            elif int(angle_detect) == 270 and h > w:
                new_angle = -90.0
            else:
                pass
        # This branch is chosen if Pytesseract unable to detect orientation
        elif type(angle_detect) == list and h > w:
            new_angle = -180.0

        return new_angle

    def rotate_img(self, img, angle):
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        twoD_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = np.abs(twoD_matrix[0, 0])
        sin = np.abs(twoD_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        twoD_matrix[0, 2] += (new_w // 2) - center[0]
        twoD_matrix[1, 2] += (new_h // 2) - center[1]

        rotated_img = cv2.warpAffine(
            img,
            twoD_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated_img

    def check_rotated_img(self, grayed_img, angle):
        extracted_text = []
        # Check PyTesseract to determine the orientation
        new_angle = self.determine_orientation_angle(grayed_img, angle)
        rotated_img = None

        if int(new_angle) is not None:
            rotated_img = self.rotate_img(grayed_img, new_angle)
            sharpened_img = self.sharpen_img(rotated_img)
            text = pytesseract.image_to_string(sharpened_img)
            text = text.replace("\n", "").split()
            for i in text:
                if i.lower() in self.english_vocab:
                    extracted_text.append(i)

        # Use original angle instead
        if len(extracted_text) < 15:
            rotated_img = self.rotate_img(grayed_img, angle)
        return rotated_img
