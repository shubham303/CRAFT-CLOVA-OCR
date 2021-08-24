import json
import os

from tqdm import tqdm

import configuration
from document_processors.PAN.NewPANCardProcessor import NewPANCardProcessor
from exception.exceptions import CannotProcessImageError
from indic_ocr.utils.image import get_all_images, load_image


def processPanCardImageFolder(config,pan_clova_config, input_path):
	
	if not configuration.output_folder:
		configuration.output_folder = os.path.join(input_path, 'document_processors/PAN/ocr_pan_output')
	os.makedirs(configuration.output_folder, exist_ok=True)
	
	images = get_all_images(input_path)
	
	new_pan_card_prcessor = NewPANCardProcessor()
	new_pan_card_prcessor.load_model(config, pan_clova_config)
	
	result = []
	for img_path in tqdm(images, unit=' images'):
		try:
			# "CLPPR4304A_MOHAN VASANT RANDIVE_SHUBHAM MOHAN RANDIVE_03051996_1" not in os.path.basename(img_path):
			#	continue
			img = load_image(img_path)
			print("processing image:", img_path)
			output = new_pan_card_prcessor.processPanCardImage(img, img_path)
			output["image_path"] = img_path
			result.append(output)
		except CannotProcessImageError as e:
			print(e)
			output= {}
			output["image_path"] = img_path
			output["error"] = str(e)
			result.append(output)
	
	return result

 
if __name__ == '__main__':
	import fire
	output = fire.Fire(processPanCardImageFolder)
	with open("{}/prediction.json".format(configuration.output_folder),"w") as f:
		json.dump(output,f)
	
	