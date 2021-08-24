
#detect language of
class ClovaModelRepository():
	
	def __init__(self, recognisers):
		self.recognisers = recognisers
	
	def __detectLanguage(self, image):
		return "en"
	
	def recognize(self, img):
		recogniser = self.recognisers[self.__detectLanguage(img)]
		return recogniser.recognize(img)
	
	