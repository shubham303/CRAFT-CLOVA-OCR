from abc import ABC, abstractmethod
import numpy as np
import imageio
import cv2
from indic_ocr.detection import Detector_Base

class End2EndOCR_Base(ABC):
    @abstractmethod
    def __init__(self, langs):
        pass
    
    @abstractmethod
    def run(self, img):
        return []

    def draw_bboxes(self, img, bboxes, out_img_file):
        rect_color = (0,0,255)
        for bbox in bboxes:
            pts = np.array(bbox["points"], np.int32).reshape((-1,1,2))
            img = cv2.polylines(img, [pts], True, rect_color)
            # TODO: Draw text near box
        # cv2.imshow(out_img_file, img)
        # cv2.waitKey(0); cv2.destroyAllWindows()
        cv2.imwrite(out_img_file, img)
        return img
