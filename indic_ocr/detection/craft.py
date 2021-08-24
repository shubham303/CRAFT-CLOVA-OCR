import numpy as np
from craft_text_detector import (
	load_craftnet_model,
	load_refinenet_model,
	get_prediction,
	empty_cuda_cache
)

from exception.exceptions import CannotProcessImageError
from indic_ocr.detection import Detector_Base


class CRAFT_Detector(Detector_Base):
	
	def __init__(self, args):
		#self.refine_net = load_refinenet_model(cuda=args.get('cuda', False))
		self.craft_net = load_craftnet_model(cuda=args.get('cuda', False))
		self.args = args
		self.refine_net = None
		
		
	def detect(self, img):
		prediction_result = get_prediction(image=np.array(img),
		                                   craft_net=self.craft_net, refine_net=self.refine_net, **self.args)
		
		# if prediction result is empty then its already a python list. calling list funtion over it results in
		# exception. otherwise its a numpy array.
		if len(prediction_result) >0:
			return [{
				'type': 'text',
				'points': points
			} for points in prediction_result['boxes'].tolist()]
		else:
			raise CannotProcessImageError("empty bounding box list returned by detector during Image preprocessing")
		
