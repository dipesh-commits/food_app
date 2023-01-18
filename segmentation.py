
try:
    import matplotlib.pyplot as plt
except:
    pass


import config
import numpy as np
import cv2
import uuid

# import warnings
# warnings.filterwarnings("error")

from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo

from loguru import logger
from PIL import Image


thing_classes = ['country_fries',
                        'tomato',
                        'apple',
                        'banana',
                        'pizza_with_ham_with_mushrooms_baked',
                        'rice',
                        'bread',
                        'chicken']
thing_colors = [(102,255,102), (102,255,255), (102,102,255),(102,150,255),(150,100,220),(101,210,222),(123,234,212),(134,111,222)]
thing_dataset_id_to_contiguous_id={100059: 0,
                                            100089: 1,
                                            100130: 2,
                                            100133: 3,
                                            101170: 4,
                                            101197: 5,
                                            101243: 6,
                                            101308: 7}

class DatasetLabels:
    TRAIN = "dataset_train"
    VAL = "dataset_val"

MetadataCatalog.get(DatasetLabels.VAL).set(thing_classes=thing_classes)
MetadataCatalog.get(DatasetLabels.VAL).set(thing_colors=thing_colors)
MetadataCatalog.get(DatasetLabels.VAL).set(thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id)

def get_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(config.MODEL_CONF)
    return cfg

def predict(img: np.array):
    logger.info(f"Getting the configuration for 2D segmentation...")
    conf = get_predictor()
    predictor = DefaultPredictor(conf)
    predictions = predictor(img)
    return predictions

def mask(img: np.array, predictions, binary=False, debug=True) -> np.array:
    all_masks = []
    mask = predictions.get("instances").get("pred_masks").to("cpu")
    mask = np.asarray(mask)
    # try:
    #     logger.debug(mask.shape)
    for i in mask:
        logger.error(i)

        item = i

        mask_arr = Image.fromarray((item*255).astype("uint8"))
        
        if binary:
            mask_arr = mask_arr.point(lambda p:1 if p>100 else 0)
            return mask_arr
        with img as img_seg:
            img_seg.load()
        blank_img = Image.new('RGB', img_seg.size)
        segmented_img = Image.composite(img_seg, blank_img, mask_arr)
        all_masks.append(segmented_img)
        if debug:
            plt.imshow(segmented_img)
            plt.show()
    # except:
    #     all_masks = [Image.fromarray(np.zeros([img.size[0], img.size[1], 3], dtype=np.uint8))]
    return all_masks

def predict_bbox(img: np.array, predictions):
    val_metadata = MetadataCatalog.get(DatasetLabels.VAL)
    bboxes = predictions.get("instances").pred_boxes
    pred_classes = predictions.get("instances").pred_classes
    confidence_scores = predictions.get("instances").scores.to("cpu")
    v = Visualizer(
        img[:,:,::-1],
        metadata = val_metadata,
        scale = 0.8
        )
    results = {}
    for idx, box in enumerate(bboxes.to("cpu")):
        class_idx = pred_classes[idx]
        class_label = val_metadata.thing_classes[class_idx]
        class_confidence_score = float('%.3g' % confidence_scores[idx])
        label_with_confidence = str(class_label) + " " + str(round(class_confidence_score*100,2)) + '%'
        v.draw_box(box)
        height, width = box[3] - box[1], box[2] - box[0]
        pos = ((box[0]+width/2).numpy(), (box[1]).numpy())
        v.draw_text(str(label_with_confidence), pos, font_size=30)
        results["output"] = {"class":class_label, "confidence":str(round(class_confidence_score*100,2))+'%',"position":pos}
    v = v.get_output()
    final_img = v.get_image()
    return final_img, results


