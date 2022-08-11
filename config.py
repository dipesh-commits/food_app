import os

SEGMENTATION_NETWORK = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
SEGMENTATION_MODEL = "models/model_final.pth"
MODEL_CONF = "models/mycfg.yaml"
NUM_WORKERS = 2
NUM_CLASSES = 8
BATCH_SIZE = 2
ROI_THRESHOLD = 0.5


DATA_DIR = "data"
APPLE_DATA_DIR = "data/apple"
BANANA_DATA_DIR = "data/banana"


MESH_FOLDER = os.path.join(APPLE_DATA_DIR, "images")
MESH_MASKS = os.path.join(APPLE_DATA_DIR, "mask")
OUTPUT_FOLDER = os.path.join(APPLE_DATA_DIR, "output_folder")
TEMP_PATH = os.path.join("sculptor", "tmp")
OUTPUT_FILE = "output.ply"
WEIGHT = 0.5

OBJ_FILE_PATH = os.path.join("static","scripts","export.obj")