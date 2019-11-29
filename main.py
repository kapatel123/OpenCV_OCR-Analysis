# Main Method - OCR
import numpy as np
import pytesseract
from pathlib import Path
from preprocess_steps import TextPreProcessor
from PIL import Image


class MainOCRProcessor(TextPreProcessor):
    def apply_preprocessor(self, pdf_images):
        if pdf_images is not None:
            for page in pdf_images:
                # convert image to array
                doc_array = np.array(page, dtype=np.uint8)
                grayed_image = self.img_to_gray(doc_array)
                edges_array = self.blur_img(grayed_image)
                init_contours = self.bw_binary_img(edges_array)
                max_contours = self.find_max_contours(init_contours)
                skew_angle = self.draw_box(max_contours)
                rotated_image = self.check_rotated_img(grayed_image, skew_angle)
                sharpened_image = self.sharpen_img(rotated_image)
            return sharpened_image
        else:
            raise ValueError("Invalid data provided.  Must be list of image(s)")

    def ocr_extractor(self, sharpened_image_list):
        docs_text = []
        rotated_images_array = []
        text = str(pytesseract.image_to_string((sharpened_image_list))).replace(
            "\n", " "
        )
        docs_text.append(text)
        rotated_images_array.append(Image.fromarray(sharpened_image_list))

        return docs_text, rotated_images_array


if __name__ == "__main__":
    pathlist = Path("testFiles").glob("*.pdf")
    filepath = [str(x) for x in pathlist]
    # doc = TextPreProcessor.pdf2image_converter_from_path(filepath)
    print(filepath)
