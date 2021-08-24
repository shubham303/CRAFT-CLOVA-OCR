class CannotProcessImageError(Exception):
	def __init__(self, img_path, message= " could not process image. image quality low"):
		self.message=message
		self.img_path = img_path
	
	
	def __str__(self):
		return f'{self.img_path} -> {self.message}'
	