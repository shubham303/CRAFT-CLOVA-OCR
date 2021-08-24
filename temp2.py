#coding=utf-8
import os
from functools import reduce
from glob import glob

import numpy as np
import cv2
import math
from PIL import Image
from tqdm import tqdm

import configuration
from indic_ocr.utils.image import saveImage
from indic_ocr.utils.img_preprocess import AutoDeskewer, AutoCrop

IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def resize(image, height):
    if image.shape[0] <= height: return image
    ratio = round(height / image.shape[0], 3)
    width = int(image.shape[1] * ratio)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite('output/resized.jpg', resized)
    return resized
def get_all_images(folder):
    files = glob(os.path.join(folder, '*'))
    images = []
    for extension in IMAGE_EXTENSIONS:
        images += [file for file in files if file.lower().endswith(extension)]
    return images

def draw_hough_lines( img, lines):
    hough_line_output = img

    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        n = 5000
        x1 = int(x0 + n * (-b))
        y1 = int(y0 + n * (a))
        x2 = int(x0 - n * (-b))
        y2 = int(y0 - n * (a))

        cv2.line(
            hough_line_output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

    cv2.imwrite('output/hough_line.jpg', hough_line_output)



def deskew(i,src, img_path):
    #src = cv2.imread("input1.jpg",cv2.IMREAD_COLOR)
    src = resize(src, 4000)
    #showAndWaitKey("blue1", src)
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    #gray = cv2.GaussianBlur(gray, (13, 13), 0)
    #gray = gray[:,:,0]
    showAndWaitKey("blue", gray)
    kernel = np.ones((5,5),np.uint8)
    erode_Img = cv2.erode(gray,kernel)
    eroDil = cv2.dilate(erode_Img,kernel) # erode and dilate
    showAndWaitKey("eroDil",eroDil)

    canny = cv2.Canny(eroDil,50,150) # edge detection
    showAndWaitKey("canny", canny)

   # contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   # approx = cv2.approxPolyDP(contours, 0.01 * cv2.arcLength(contours, True), True)
   # print(approx)
    #cv2.drawContours(src, contours, -1, (0,255, 0) , 3)
   # showAndWaitKey("contours", src)
    print(math.pi / 2)
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 90, minLineLength=50, maxLineGap=10)  # Hough Lines Transform
    drawing = np.zeros(src.shape[:], dtype=np.uint8)

    maxY = 0
    degree_of_bottomline = 0
    index = 0
    degrees=[]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        k = float(y1 - y2) / (x1 - x2)
        degree = np.degrees(math.atan(k))
        if(degree < 45 and degree >-45):
            degrees.append(degree)
        index = index + 1
    showAndWaitKey("houghP", drawing)
    positive_count = sum(map(lambda x: x>0, degrees))
    negative_count = sum(map(lambda x: x<0, degrees))

    import statistics
    dev = statistics.pstdev(degrees)
    median = statistics.median(degrees)
    
    degrees = [d  for d in degrees if abs(d-median) <= dev]
    
    rotation_angle = reduce(lambda a, b: a + b, degrees) / len(degrees)
    img = Image.fromarray(src)
    rotateImg = img.rotate(rotation_angle)
    rotateImg_cv = np.array(rotateImg)
    #cv2.imshow("rotateImg", rotateImg_cv)
    #cv2.imwrite("output/{}.jpg".format(os.path.basename(img_path)), rotateImg_cv)
    #cv2.waitKey()

    gray = cv2.cvtColor(rotateImg_cv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    erode_Img = cv2.erode(gray, kernel)
    eroDil = cv2.dilate(erode_Img, kernel)  # erode and dilate
    showAndWaitKey("eroDil2", eroDil)

    canny = cv2.Canny(eroDil, 50, 150)  # edge detection
    showAndWaitKey("canny2", canny)

    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 90, minLineLength=200, maxLineGap=10)  # Hough Lines Transform
 
    drawing = np.zeros(src.shape[:], dtype=np.uint8)

    maxY = 0
    degree_of_bottomline = 0
    index = 0
    degrees = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        k = float(y1 - y2) / (x1 - x2)
        degree = np.degrees(math.atan(k))
        if (degree < 45 and degree > -45):
            degrees.append(degree)
        index = index + 1
    showAndWaitKey("houghP2", drawing)
    return  rotateImg_cv


def showAndWaitKey(winName, img):
    #cv2.imshow(winName, img)
    cv2.imwrite("output/{}.jpg".format(winName), img)
    #cv2.waitKey()


if __name__ == "__main__":
    images = get_all_images("input_images")
    result = []
    for i, img_path in enumerate(images) :
        if "IRUHK1321O_PAQAH JEKEX KUBUHU_ZOXEY PUMOVU HEDUJE_31062003" not in os.path.basename(img_path):
            continue
        configuration.debug=True
        configuration.output_folder= "ocr_pan_output"
        src = cv2.imread(img_path)
        deskew = AutoDeskewer()
        crop = AutoCrop()
        img = crop.process(src, img_path)
        img = deskew.process(img, img_path)
        img = crop.process(img, img_path)
        saveImage(img, img_path, "")