import json
import re

import numpy as np


class PANCardProcessor:
	
	
	def extract_info(self, bbox_list, gt_data=None, ):
		printed_text_income=[]
		printed_text_signature= []
		printed_text_pan_text=[]
		pancard_fields = {}
		
		for box in bbox_list:
			if box["type"] == "text":
				box["points"] = np.array(box["points"]).ravel().tolist()
				
				if ("income" in box["text"].lower()):
					printed_text_income = box
				
				if ("/" in box["text"].lower()):
					dob = box
					bbox_list.remove(box)
				
				if ("permanent" in box["text"].lower()):
					printed_text_pan_text = box
				if ("signature" in box["text"].lower()):
					printed_text_signature = box
		
		name = []
		father = []
		pan_number = None
		signature =[]
		
		bbox_list = sorted(bbox_list, key=lambda x: x["points"][1])
		
		for box in bbox_list:
			if "/" in box["text"]:
				dob = box
				continue
				
			if box["type"] == "text":
				if box["points"][1] > printed_text_income["points"][7] and box["points"][7] < printed_text_pan_text[
					"points"][1]:
					if len(name)>0:
						if self.box_in_same_line(name, box):
							name.append(box)
						else:
							father.append(box)
					else:
						name.append(box)
				
				if box["points"][1] > printed_text_pan_text["points"][7] and box["points"][7] < printed_text_signature[
					"points"][1]:
					if not pan_number:
						pan_number = box
					else:
						signature = box
				
			
		
		pancard_fields["name"] = {}
		pancard_fields["father"] = {}
		pancard_fields["dob"] = {}
		pancard_fields["pan_number"] = {}
		
		pancard_fields["name"]["text"] = self.combine_bounding_boxes(name)
		pancard_fields["father"]["text"] = self.combine_bounding_boxes(father)
		pancard_fields["dob"]["text"] = dob["text"]
		pancard_fields["dob"]["text"] = self.rectifydob(list(pancard_fields["dob"]["text"]))
		pancard_fields["pan_number"]["text"] = pan_number["text"]
		
		if gt_data:
			pancard_fields["name"]["confidence"] = name[0]["confidence"]
			pancard_fields["father"]["confidence"] = father[0]["confidence"]
			pancard_fields["dob"]["confidence"] = dob["confidence"]
			pancard_fields["pan_number"]["confidence"] = pan_number["confidence"]
		
		print(pancard_fields)
		return pancard_fields

