from abc import ABC, abstractmethod

from indic_ocr.recognition.ClovaModelRepository import ClovaModelRepository


class RecognizerBase(ABC):
    
    @abstractmethod
    def __init__(self, langs):
        pass
    
    @abstractmethod
    def recognize(self, img):
        pass

def load_recognizer(config):
    recognizer_cfgs = config['recognizer']
    recognisers={}
    for recognizer_cfg in recognizer_cfgs:
        if recognizer_cfg['name'] == 'clova_ai':
            from indic_ocr.recognition.clova_ai import ClovaAI_Recognizer
            ClovaAI_Recognizer = ClovaAI_Recognizer(recognizer_cfg['lang'],
                                      gpu=config.get('gpu', False),
                                      model_dir=recognizer_cfg.get('model_dir', None),
                                      model_name= recognizer_cfg.get('model_name', "best_accuracy.pth"),
                                      args=recognizer_cfg.get('params', {}))
            recognisers[recognizer_cfg["lang"]] = ClovaAI_Recognizer
            
        elif recognizer_cfg['name'] == 'clova_ai_pan':
            from indic_ocr.recognition.clova_ai_pan import ClovaAI_Pan_Recognizer
            ClovaAI_Recognizer = ClovaAI_Pan_Recognizer(recognizer_cfg['lang'],
                                                        gpu=config.get('gpu', False),
                                                        model_dir=recognizer_cfg.get('model_dir', None),
                                                        model_name= recognizer_cfg.get('model_name', "best_accuracy.pth"),
                                                        args=recognizer_cfg.get('params', {}))
            recognisers[recognizer_cfg["lang"]] = ClovaAI_Recognizer
        else:
            print('No support for recognizer:',recognizer_cfg['name'])
            raise NotImplementedError
    return ClovaModelRepository(recognisers)