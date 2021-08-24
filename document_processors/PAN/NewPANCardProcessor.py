import json
import os
import re

import cv2
import imageio
import numpy as np
from tqdm import tqdm

import configuration
from document_processors.PAN.utils import isTextdate, is_box_in_same_line_in_pan_card, get_combined_text_from_box_list, \
	get_combined_text_from_box_list_dob, rectifydob
from exception.exceptions import CannotProcessImageError
from indic_ocr.ocr import OCR
from indic_ocr.utils.image import crop_image_using_quadrilateral, get_all_images

PAN_JSON_CONFIG = "config/clova_pan.json"
PAN_NUMBER_LENGTH = 10

class NewPANCardProcessor:
	
	def load_model(self, config, pan_clova_config= PAN_JSON_CONFIG):
		self.ocr= OCR(config)
		from indic_ocr.recognition import load_recognizer
		with open(pan_clova_config, encoding='utf-8') as f:
			rec_config = json.load(f)
		
	
		self.pan_number_ocr = load_recognizer(rec_config)

	def extract_info(self,bbox_list,gt_data=None):
		
		printed_text_card=None
		printed_text_name=None
		printed_text_father=None
		printed_text_date = None
		
		
		dob=None
		
		
		for box in bbox_list:
			box["points"] = [j for sub in box["points"] for j in sub]
			
		bbox_list = sorted(bbox_list, key=lambda x: x["points"][1])
		
		for box in bbox_list:
			if box["type"] =="text":
					#TODO try using fuzzywoozy package for text matching.
					
					
					if ("card" in box["text"].lower()):
						printed_text_card = box
					
					if  "name" in box["text"].lower() and not printed_text_name:
						printed_text_name = box
					
					if  "father" in box["text"].lower():
						printed_text_father = box
					
					if "date" in box["text"].lower():
						printed_text_date = box
				
		
		
		
		if not printed_text_date:
			raise CannotProcessImageError("printed  text date not recognised")
		
		if not printed_text_card:
			raise CannotProcessImageError ("printed_text_card not recognised")
		
		if not printed_text_name:
			raise CannotProcessImageError ("printed_text_name not recognised")
		
		if not printed_text_father:
			raise CannotProcessImageError("printed_text_father not recognised")
		
	
		
		
		
		name = []
		father = []
		pan_number = None
		pancard_fields={}
		dob=[]
		
		
	
		for box in bbox_list:
			
			if not pan_number and box["points"][1] > printed_text_card["points"][7] and box["points"][0] < \
					printed_text_card["points"][0]:
				pan_number =box
				
			if printed_text_name["points"][7] < box["points"][1] and printed_text_father["points"][1] > box[
				"points"][7]:
				name.append(box)
				
			if printed_text_father["points"][7] < box["points"][1] and printed_text_date["points"][1] > box[
				"points"][7] and printed_text_card["points"][4] > box["points"][4]:
				father.append(box)
				
			if printed_text_date["points"][7] < box["points"][1] and printed_text_date["points"][0] > box[
				"points"][0]:
				dob.append(box)
		
		if not pan_number:
			raise CannotProcessImageError("pan_number not recognised")
		if not father:
			raise CannotProcessImageError("father not recognised")
		if not name:
			raise CannotProcessImageError("name not recognised")
		if not dob:
			raise CannotProcessImageError("dob not recognised")
		
		
		pancard_fields["pan_number"] = pan_number
		pancard_fields["father"] = father
		pancard_fields["name"] = name
		pancard_fields["dob"] = dob
		
		return pancard_fields

	def processPanCardImage(self,img,img_path):

		#out_file_1 = os.path.join(output_folder, os.path.splitext(os.path.basename(input_path))[0])
		
	
		pred_data, img = self.ocr.process_img(img, img_path)
		pred_data = pred_data["data"]
		
		pancard_fields = self.extract_info(pred_data)
		
		if pancard_fields:
			pan_number_box = pancard_fields["pan_number"]
			img_crop = crop_image_using_quadrilateral(img, np.reshape(np.array(pan_number_box['points']), (-1,2)))
			result = self.pan_number_ocr.recognize(img_crop)
			pancard_fields["pan_number"]["text"]=result["text"]
			
			#return and save just predicted boxes.
			pancard_fields["pan_number"] = pancard_fields["pan_number"]["text"]
			pancard_fields["father"] = get_combined_text_from_box_list(pancard_fields["father"])
			pancard_fields["name"] =get_combined_text_from_box_list(pancard_fields["name"])
			pancard_fields["dob"] =rectifydob(get_combined_text_from_box_list_dob(pancard_fields["dob"]))
			return pancard_fields
		
		else:
			raise CannotProcessImageError(img_path)
	
