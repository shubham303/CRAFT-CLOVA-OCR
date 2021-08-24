import json
import re

from exception.exceptions import CannotProcessImageError


def isTextdate(text):
	# function to identify whether given text is date or something else
	return len(text) == 10  and "/" in text

def rectifydob(dob_text):
	# sometimes for date, recogniser predicts "/" for 1 and vice a versa.  we know for PAN card, 2nd and 5th
	# character is / and rest of the characters are numbers. using that logic function rectifies the date value.
	if len(dob_text)!=10:
		raise CannotProcessImageError("cannot process image. Date of birth incorrectly predicted")
	
	if isinstance(dob_text, str):
		dob_text = list(dob_text)
	dob_text[2] = "/"
	dob_text[5] = "/"
	

	for i in range(len(dob_text)):
		if i == 2 or i == 5:
			dob_text[i] = "/"
		elif dob_text[i] == "/":
			dob_text[i] = '1'
	
	return "".join(dob_text)
	
def is_box_in_same_line_in_pan_card(box1, box2):
	# check boxes in same line, assumption is document is horizontally oriented.
	if box1["points"][1] <= box2["points"][1] and box1["points"][7] > box2["points"][1]:
		return True
	
	if box1["points"][1] <= box2["points"][7] and box1["points"][7] > box2["points"][7]:
		return True
	
	if box1["points"][1] >= box2["points"][1] and box1["points"][7] < box2["points"][7]:
		return True
	
	return False

def is_box_in_same_line_in_pan_card( box1, box2):
	return is_box_in_same_line_in_pan_card(box1, box2) or is_box_in_same_line_in_pan_card(box2, box1)
	
def remove_unnecessary_boxes( bbox_list):
	
	remove_list = ["आयकर", "विभाग", "भारत", "सरकार", "INCOME", "TAX", "DEPARTMENT", "GOVT.", "GOVT", "OF", "INDIA",
	               "सत्यमेव",
	               "जयते", "स्थायी", "लेखा", "संख्या", "कार्ड", "Account", "Number", "Name", "का", "Father", "Name",
	               "की", "तारीख", "Date", "of", "Birth", "हस्ताक्षर", "Signature", ]
	
	for l in remove_list:
		for box in bbox_list:
			if re.search(l.lower(), box["text"]) or re.search(box["text"], l.lower()):
				bbox_list.remove(box)
	
	return bbox_list


#TODO remove this function.
def get_combined_text_from_box_list(bbox_list):
	
	bbox_list = sorted(bbox_list, key=lambda x: x["points"][0])
	
	text = [box["text"] for box in bbox_list]
	
	return " ".join(text)

def get_combined_text_from_box_list_dob(bbox_list):
	bbox_list = sorted(bbox_list, key=lambda x: x["points"][0])
	
	text = [box["text"] for box in bbox_list]
	
	return "".join(text)