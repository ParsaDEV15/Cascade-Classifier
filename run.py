from ModelsCode.YOLODetector import ObjectExtractor
from ModelsCode.CascadeClassifier import CascadeTrainer

POS_DIR = 'data/Positive'
NEG_DIR = 'data/Negative'
NEW_DIR = 'Dataset'

if __name__ == '__main__':
    detector = ObjectExtractor(POS_DIR, NEG_DIR, NEW_DIR)
    detector.extract()

    trainer = CascadeTrainer(NEW_DIR, 'Phone')
    trainer.train()