# -*- coding: utf-8 -*-
import pdf2image
import time,os,cv2
from pdf2image import convert_from_path, convert_from_bytes
import tempfile
import numpy as np
import pandas as pd
import pytesseract
import Image2Table
import shutil
import imutils

from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

def CreateFolder(Folder_Name_With_Path):
    if not os.path.exists(Folder_Name_With_Path):
        os.mkdir(Folder_Name_With_Path)

def box_extraction(PDF_FILE_PATH_for_box_extraction):
    print(PDF_FILE_PATH_for_box_extraction)
    img = cv2.imread(PDF_FILE_PATH_for_box_extraction, 0)  # Read the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40
     
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, 0)

    # Find contours for image, which will detect all the boxes
    contours = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[0:5]
    if not hasattr(box_extraction, "counter"):
        box_extraction.counter = 0  # it doesn't exist yet, so initialize it

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 5000 and h < 6000 and w >2500 and h > 200):
            box_extraction.counter += 1
            tbl_img = img[y:y+h, x:x+w]
            cv2.line(tbl_img, (0, 0), (w, 0), (0,0,0), 30)
            cv2.line(tbl_img, (0, h), (w, h), (0,0,0), 30)
            cv2.imwrite('tbl_img'+str(box_extraction.counter)+'.png', tbl_img)


def Pdf_to_Text(pdf_file):
    pdf_file = PDF_PATH+pdf_file
    PDF_FILE_PATH = pdf_file.split(".")[0]
    CreateFolder(PDF_FILE_PATH)
    os.chdir(PDF_FILE_PATH)
    pdf2image.convert_from_path(pdf_file,output_folder=PDF_FILE_PATH,thread_count=6,dpi=600,fmt='ppm')
    df = pd.DataFrame([]) 
    for img in os.listdir():
        if img.endswith('.ppm'):
            #print(os.getcwd())
            box_extraction(img)
    for img2 in os.listdir():
        if img2.endswith('.png') and img2.startswith('tbl_img'):
            df = df.append(Image2Table.Table_Extraction(img2), ignore_index = True)
    os.chdir(PDF_PATH)
    PDF_FILE = PDF_FILE_PATH.split('/')[-1]
    df.to_excel(PDF_FILE+".xlsx")
    shutil.rmtree(PDF_FILE_PATH)
    return df