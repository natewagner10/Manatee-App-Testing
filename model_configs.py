"""
Mote Marine Laboratory Collaboration

Manatee Matching Program

Written by Nate Wagner
"""

from Documents.Mote_Manatee_Project.AI.maskrcnn_files.Mask_RCNN.mrcnn.config import Config

class manateeAIConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "scar_finder"      
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 3    
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.50    
    MAX_GT_INSTANCES = 120    
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)    
    LOSS_WEIGHTS = {'rpn_class_loss': 1.0,
                    'rpn_bbox_loss': 1.0,
                    'mrcnn_class_loss': 1.0,
                    'mrcnn_bbox_loss': 1.0,
                    'mrcnn_mask_loss': 1.0}
    LEARNING_RATE = 0.001   
    MEAN_PIXEL = [248.56, 248.56, 248.56]
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_NMS_THRESHOLD=0.99  

