import json
import os
from glob import glob

import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

from document_processors.PAN.NewPANCardProcessor import NewPANCardProcessor
from indic_ocr.utils.img_preprocess import AutoDeskewer


def extract_ground_truth(gt):
    gt= gt["data"]
    pancard_fields = {}
    pancard_fields["name"]=[]
    pancard_fields["father"]=[]
    for entry in gt:
        if "entity" in entry:
            if entry["entity"] == "id":
                pancard_fields["pan_number"] = entry["text"]
            if entry["entity"] == "date_of_birth":
                pancard_fields["dob"] = entry["text"]
                
            if entry["entity"] == "name":
                pancard_fields["name"].append(entry["text"])
    
            if entry["entity"] == "parent_name":
                pancard_fields["father"].append(entry["text"])
    l=pancard_fields["name"]
    pancard_fields["name"] =" ".join(str(x) for x in l)
    
    l = pancard_fields["father"]
    pancard_fields["father"] = " ".join(str(x) for x in l)
    
    return pancard_fields


def run(config_json, input_path, output_folder=None, preprocessors=["deskew"],gt_folder=None):
    
    preprocessor=None
    if preprocessors:  # Load pre-processor
        if type(preprocessors) == str:
            preprocessors = [preprocessors]
        
        from indic_ocr.utils.img_preprocess import PreProcessor
        preprocessor = PreProcessor(preprocessors)
        
   
    pancardProcessor = NewPANCardProcessor()
    pancardProcessor.load_model(config_json)
    result = pancardProcessor.processPanCardImageFolder(input_path, output_folder, preprocessor)
    #print(result)
    
    with open('input_images/text.txt','w') as f:
        f.write(json.dumps(result))

    with open('input_images/text.txt',"r") as f:
        data = f.read()
    result = json.loads(data)
    
    correct_name_count = 0
    correct_father_name_count = 0
    correct_pn_count = 0
    correct_dob_count = 0
    correct_doc_count = 0
    both =0
    gts = glob(os.path.join(input_path, '*.json'))

    for gt in gts:
        gt_data = json.load(open(gt))
        gt_file_name = os.path.basename(gt)
        gt_data = extract_ground_truth(gt_data)
    
    
        for item in result:
            if ( item["image_path"].replace(".jpg",".json") == gt):
                pred_data =item
                break
        try:
            if pred_data["name"]["text"].replace(".","") == gt_data["name"].replace(".",""):
                correct_name_count += 1
            else:
                print("actual name: {} predicted: {} filename:{} ".format(gt_data["name"], pred_data[
                    "name"]))
            if pred_data["father"]["text"] == gt_data["father"]:
                correct_father_name_count += 1
        
            if pred_data["pan_number"]["text"] == gt_data["pan_number"]:
                correct_pn_count += 1
            
            if pred_data["dob"]["text"] == gt_data["dob"]:
                correct_dob_count += 1
        
            if pred_data["father"]["text"] == gt_data["father"] and pred_data["name"]["text"].replace(".","") == gt_data["name"].replace(".",""):
                both += 1
        except:
            print("error")
    print("correct_names {}  correct fathers {} correct pan {} correct dob {}".format(correct_name_count,
                                                                                      correct_father_name_count,
                                                                                      correct_pn_count,
                                                                                      correct_dob_count))

    print("both correct {}".format(both))
    


if __name__ == '__main__':
    import fire
    
    fire.Fire(run)
