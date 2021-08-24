import json
import os

from tqdm import tqdm

from indic_ocr.utils.image import get_all_images


file  = open("/home/shubham/Documents/MTP/Craft-Clova-OCR/output/prediction.json","r")
data = json.load(file)

correct_pan_count =0
correct_name_count =0
correct_father_count=0
correct_dob_count=0
all_correct =0

document_reported_error=0
total =0
for d in data:
	total+=1
	gt_file_name = d["image_path"]
	gt = os.path.basename(d["image_path"]).replace(".jpg", "").split("_")
	temp=0
	cont=False
	
	if cont:
		total-=1
		continue
	if "error" in d:
		document_reported_error+=1
		print(d["error"] ,"  ",  d["image_path"])
		continue
		
	if d["pan_number"].lower()[0:5] == gt[0].lower()[0:5]:
		correct_pan_count+=1
		temp+=1
	
	
	
	if d["father"].lower() == gt[1].lower():
		correct_father_count += 1
		temp += 1
	#else:
	#	print("predicted:", d["father"], "actual pan: ", gt[1], "image:", d["image_path"])
		
	if d["name"].lower() == gt[2].lower():
		correct_name_count += 1
		temp += 1
	
	
	if d["dob"].replace("/","") == gt[3].replace("/",""):
		correct_dob_count += 1
		temp+=1
	
	#model makes error in predicting J and U. // need to train on this.
	# also makes error in 1 and 4
	if temp ==4:
		all_correct +=1

print("total documents ",total)
print("correct pan", correct_pan_count)
print("correct name",correct_name_count)
print("correct father",correct_father_count)
print("correct dob",correct_dob_count)
print("all correct",all_correct)
print("error reported ",document_reported_error)