class DocumentProcessor:
	def __init__(self, documentProcessors):
		self.documentProcessors = documentProcessors
		
	#get appropriate document processor based on data stored inside it.
	def getDocumentProcessor(self, json_data):
		return self.documentProcessors[self.__getTypeOfImageDocument(json_data)]
	
	def __getTypeOfImageDocument(self, json_data):
		#TODO replace this code by function call to module which recognises a particular type of doc.
		return "pan"