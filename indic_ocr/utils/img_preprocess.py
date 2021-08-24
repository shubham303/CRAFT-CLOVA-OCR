import json
import math
import os
from functools import reduce

import cv2
from PIL import Image
import numpy as np
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray

import configuration
from indic_ocr.utils.image import crop_image_using_quadrilateral, saveImage


class PreProcessor:
    def __init__(self, processors_list):
        self.processors = []
        for processor_name in processors_list:
            processor = None
            if processor_name == 'deskew':
                processor = AutoDeskewer()
            if processor_name == "crop":
                processor = AutoCrop()
            if processor:
                self.processors.append(processor)
            else:
                print('Unknown pre-processor: ', processor_name)
        
    def process(self, img,img_path=None):
        '''
        preprocess the image and if debug = true save the result.
        '''
        for processor in self.processors:
            img = processor.process(img, img_path)
        
        if configuration.debug:
            out_path = os.path.join(configuration.output_folder,
                        os.path.splitext(os.path.basename(img_path))[0]) \
                        + '-preprocess_output.jpg'
            cv2.imwrite(out_path, img)
        
        return img
    

from abc import ABC, abstractmethod
class PreProcessorBase(ABC):
    @abstractmethod
    def process(self, img, img_path = None):
        '''
        Assumes cv2 image as input
        '''
        pass
    

class AutoDeskewer(PreProcessorBase):
    """
    deskew image automatically. if debug is set to True, then save deskew result in file.
    """
    def process(self, img, img_path = None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((configuration.IMG_ERODE_KERNEL_SIZE, configuration.IMG_ERODE_KERNEL_SIZE), np.uint8)
        erode_Img = cv2.erode(gray, kernel)
        eroDil = cv2.dilate(erode_Img, kernel)  # erode and dilate
        if configuration.debug:
            saveImage(eroDil, img_path, "erodil")
    
        canny = cv2.Canny(eroDil, configuration.EDGE_LOWER_THRESHOLD, configuration.EDGE_UPPER_THRESHOLD)  # edge detection

        if configuration.debug:
            saveImage(canny, img_path, "canny")
            
        lines = cv2.HoughLinesP(canny, configuration.RHO, configuration.THETA, configuration.MIN_VOTE_HOUGH,
                                minLineLength=configuration.MIN_LINE_LENGTH,
                                maxLineGap=configuration.MAX_LINE_GAP)  #
        

        # Hough
        # Lines Transform
        index = 0
        degrees = []
        filtered_lines=[]
        drawing = np.zeros(img.shape[:], dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            k = float(y1 - y2) / (x1 - x2)
            degree = np.degrees(math.atan(k))
            #remove lines of angle more than 45 degree both clockwise and anticlockwise
            if (degree < 45 and degree > -45):
                degrees.append(degree)
                filtered_lines.append(line)
            index = index + 1
            
      #  import statistics
    #dev = statistics.pstdev(degrees)
     #   median = statistics.median(degrees)
        
        X= []

        for angle in range(-45, 45, 3):
            l= angle
            r = angle+3
            temp_list = [[d,f] for d, f in zip(degrees, filtered_lines) if l <=d<=r]
            temp = len(temp_list)
            if temp > len(X):
                X = temp_list

        degrees = [ x[0] for x in X]
        filtered_lines = [x[1] for x in X]
        

        if configuration.debug:
            for line in filtered_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
            saveImage(drawing, img_path, "hough")
        if len(degrees)>0:
            rotation_angle = reduce(lambda a, b: a + b, degrees) / len(degrees)
        else:
            rotation_angle = 0
        img = Image.fromarray(img)
        rotateImg = img.rotate(rotation_angle)
        rotateImg_cv = np.array(rotateImg)
        
        if configuration.debug:
            saveImage(rotateImg_cv, img_path, "deskew")
        return rotateImg_cv

class AutoCrop(PreProcessorBase):
    def __init__(self):
        from indic_ocr.detection import load_detector
        with open("config/craft.json", encoding='utf-8') as f:
            det_config = json.load(f)
        self.detection_ocr = load_detector(det_config)
        
    def process(self, img, img_path):
        bbox = self.detection_ocr.detect(img)

        rect_color = (0, 0, 255)
        if configuration.debug:
            drawing = img.copy()
            for box in bbox:
                pts = np.array(box["points"], np.int32).reshape((-1, 1, 2))
                drawing = cv2.polylines(drawing, [pts], True, rect_color)
                # TODO: Draw text near box
            saveImage(drawing, img_path, "craft_box_image_crop")
        
        x_min = 100000
        y_min = 100000
        x_max = 0
        y_max = 0
        img_width = img.shape[1]
        img_height = img.shape[0]
        
        
        
        for box in bbox:
            box = box["points"]
            points = [j for sub in box for j in sub]
            x_points = points[0::2]
            y_points = points[1::2]
        
            mini_x = min(x_points)
            mini_y = min(y_points)
        
            max_x = max(x_points)
            max_y = max(y_points)
        
            x_min = min(x_min, mini_x)
            y_min = min(y_min, mini_y)
        
            x_max = max(x_max, max_x)
            y_max = max(y_max, max_y)
    
        #width and height of cropped out part
        width = x_max - x_min
        height = y_max - y_min
        
        # increase width and length of cropped image by 10 percent just to be on safe side.
        x_padding =width/30
        y_padding = height/30
       
        x_min = max(x_min-x_padding, 0)
        x_max = min( x_max+x_padding, img_width)
        y_min = max(y_min-y_padding, 0)
        y_max = min( y_max+y_padding, img_height)
     
        
        img = crop_image_using_quadrilateral(img, [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        
        if configuration.debug:
            out_file = os.path.join(configuration.output_folder, os.path.splitext(os.path.basename(img_path))[0])
            cv2.imwrite("{}_crop.jpg".format(out_file), img)
        
        return img