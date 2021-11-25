import sys, os

sys.path.append(os.path.abspath('libs/clova_ai_pan_recognition'))

import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image

from indic_ocr.recognition import RecognizerBase

#TODO avoid multiple versions clova model, try to use single model.
class ClovaAI_Pan_Recognizer(RecognizerBase):
	'''
	Inference code based on: github.com/clovaai/deep-text-recognition-benchmark/blob/master/demo.py
	'''
	
	def __init__(self, lang, gpu=False, model_dir=None, model_name=None, args={}):
		self.lang = lang
		self.device = torch.device('cuda' if gpu else 'cpu')
		self.set_options(args)
		self.load_characters(model_dir)
		self.load_model(model_dir, model_name)
	
	# opt.saved_model, opt.character
	
	def set_options(self, args):
		default_options = {
			'imgH': 32,
			'imgW': 100,
			'PAD': False,
			'num_fiducial': 20,
			'rgb': False,
			'input_channel': 1,
			'output_channel': 512,
			'hidden_size': 256,
			'num_gpu': torch.cuda.device_count()
		}
		
		for key in default_options:
			if key not in args:
				args[key] = default_options[key]
		
		from munch import munchify
		self.opt = munchify(args)
		return
	
	def load_model(self, root_dir, model_name, dataset=None):
		## --- Load Prediction component --- ##
		opt = self.opt
		if 'CTC' in opt.Prediction:
			from libs.clova_ai_pan_recognition.utils import CTCLabelConverter
			self.converter = CTCLabelConverter(opt.character)
		else:
			from libs.clova_ai_pan_recognition.utils import AttnLabelConverter
			self.converter = AttnLabelConverter(opt.character)
		self.opt.num_class = len(self.converter.character)
		
		## ---- Data loader ---- ##
		from libs.clova_ai_pan_recognition.dataset import AlignCollate
		self.AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
		if opt.rgb:
			opt.input_channel = 3
		
		## --- Load model --- #
		
		model_dir = os.path.join(root_dir, self.lang)
		if not os.path.isdir(model_dir):
			exit('ERROR: Model folder %s not found.' % model_dir)
		opt.saved_model = os.path.join(model_dir, model_name)
		
		from libs.clova_ai_pan_recognition.model import Model
		self.model = torch.nn.DataParallel(Model(opt)).to(self.device)
		self.model.load_state_dict(torch.load(opt.saved_model, map_location=self.device))
		self.model.eval()
		return
	
	#todo this is redundant function. similar character loading functionality is available in utils.py in clova text
	# recognition.
	def load_characters(self, model_dir):
		self.opt.character = []
		if 'en' == self.lang:
			import string
			# TODO add special characters like space tab newline etc refer indicOCR repo
			self.opt.character = string.printable[:-6]
			#self.opt.character = "0123456789abcdefghijklmnopqrstuvwxyz"
		if "hi" == self.lang:
			
			self.opt.character =['-', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ'] #self.opt.character = [unichr(s) for s in range(0x900, 0x980)]
		# elf.opt.character.append('\u200d')
		# self.opt.character.append('\u200c')
		
		return
	
	def inference(self, batch):
		with torch.no_grad():
			image_tensors, _ = self.AlignCollate_demo(batch)
			batch_size = image_tensors.size(0)
			image = image_tensors.to(self.device)
			# For max length prediction
			text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)
			
			if 'CTC' in self.opt.Prediction:
				preds = self.model(image, text_for_pred)
				
				# Select max probabilty (greedy decoding) then decode index to character
				preds_size = torch.IntTensor([preds.size(1)] * batch_size)
				_, preds_index = preds.max(2)
				preds_str = self.converter.decode(preds_index, preds_size)
			else:
				preds = self.model(image, text_for_pred, is_train=False)
				
				# Select max probabilty (greedy decoding) then decode index to character
				length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
				_, preds_index = preds.max(2)
				preds_str = self.converter.decode(preds_index, length_for_pred)
			
			preds_max_prob, _ = F.softmax(preds, dim=2).max(dim=2)
			result = []
			for pred, pred_max_prob in zip(preds_str, preds_max_prob):
				if 'Attn' in self.opt.Prediction:
					pred_EOS = pred.find('[s]')
					pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
					pred_max_prob = pred_max_prob[:pred_EOS]
				# calculate confidence score (= multiply of pred_max_prob)
				try:
					confidence_score = pred_max_prob.cumprod(dim=0)[-1]
				except:
					confidence_score = 0
				
				result.append((pred, float(confidence_score)))
			
			return result
	
	def recognize(self, img):
		if type(img) == np.ndarray:
			img = Image.fromarray(img).convert('L')
		result = self.inference([(img, None)])
		return {
			'text': result[0][0],
			'confidence': result[0][1]
		}
