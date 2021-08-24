import json
import os
from tqdm import tqdm
import configuration


from indic_ocr.utils.image import get_all_images, load_image


class OCR:
    def __init__(self, config_json: str,
                 additional_languages: list=None,
                 qr_scan=False):
        
        with open(config_json, encoding='utf-8') as f:
            config = json.load(f)
        
        if additional_languages is not None:
            config['langs'] = ['en'] + additional_languages
        
        self.draw = config['draw'] if 'draw' in config else False
        
        print('Loading models using', config_json)
       
        self.load_models(config)
        print('OCR Loading complete!')

    def load_models(self, config):
        from indic_ocr.detection import load_detector
        detector = load_detector(config)
        detect_only = config['recognizer']['disabled'] if 'disabled' in config['recognizer'] else False
        recognizer = None
        if not detect_only:
            from indic_ocr.recognition import load_recognizer
            recognizer = load_recognizer(config)
        
        from indic_ocr.end2end.detect_recog_joiner import DetectRecogJoiner
        self.extractor = DetectRecogJoiner(detector, recognizer)

        if configuration.preprocessors:  # Load pre-processor
            from indic_ocr.utils.img_preprocess import PreProcessor
            self.preprocessor = PreProcessor(configuration.preprocessors)
            
        return
    
    def process_folder(self, input_folder):
        
        output_folder= configuration.output_folder
        
        if not output_folder:
            output_folder = os.path.join(input_folder, 'ocr_output')
        os.makedirs(output_folder, exist_ok=True)
        images = get_all_images(input_folder)
        
        for img_path in tqdm(images, unit=' images'):
            img = load_image(img_path)
            self.process_img(img, img_path)
        
        return
    
    def process_img(self, img, img_path):

        # Pre-process image
        if self.preprocessor:
            img = self.preprocessor.process(img,img_path)
        
        
        # Check if already processed
        out_file = os.path.join(configuration.output_folder, os.path.splitext(os.path.basename(img_path))[0])
        
        if configuration.skip_if_done and os.path.isfile(out_file + '.json'):
            return out_file
        
        # Run OCR
        bboxes = self.extractor.run(img)
        
        # Save output
        if self.draw:
            img = self.extractor.draw_bboxes(img, bboxes, out_file+'.jpg')
        gt = {
            'data': bboxes,
            'height': img.shape[0],
            'width': img.shape[1],
        } # Add more metadata
        
        if configuration.debug:
            with open(out_file+'.json', 'w', encoding='utf-8') as f:
                json.dump(gt, f, ensure_ascii=False, indent=4)
            
        # return gt to extract information
        # return preprocesed img for further processing.
        return gt , img
        
        