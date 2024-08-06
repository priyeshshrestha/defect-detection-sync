import os
import cv2
import time
import traceback
from uuid import uuid4
from datetime import datetime, timedelta
from celery.signals import task_failure
from util import decode, crop_image
from db_request import update_defectPoint
from preprocess import preprocess_orange, preprocess_big, preprocess_white
from pyaml_env import parse_config

config = parse_config("config/config.yaml")

MODEL_TYPE = config["model_type"]
REFERENCE_MODELS = MODEL_TYPE + "_models"
UPLOAD_PATH = config["paths"]["uploads"]

if MODEL_TYPE == 'tf':
    from detector import DetectorTF
    model_filename = "frozen_inference_graph.pb"
    DetectorClass = DetectorTF

else:
    from detector_yolo import DetectorYOLO
    model_filename = "best.pt"
    DetectorClass = DetectorYOLO

PAD_MODEL_NAME = config[REFERENCE_MODELS]["pad"]
SMCU_MODEL_NAME = config[REFERENCE_MODELS]["smcu"]
CIRCUIT_MODEL_NAME = config[REFERENCE_MODELS]["circuit"]
SMLMA_MODEL_NAME = config[REFERENCE_MODELS]["smlma"]
ORANGE_DETECTION = config[REFERENCE_MODELS]["orange"]
BIG_DETECTION = config[REFERENCE_MODELS]["big"]
WHITE_DETECTION = config[REFERENCE_MODELS]["white"]
WHITE_SIZE_THRES = config["processor"]["preprocess"]["white_size_threshold"]
BIG_SIZE_THRES = config["processor"]["preprocess"]["big_size_threshold"]

MODEL_NAMES = {
    "pad": PAD_MODEL_NAME,
    "qfp": PAD_MODEL_NAME,
    "circuit": CIRCUIT_MODEL_NAME,
    "sm7": CIRCUIT_MODEL_NAME,
    "smlma": SMLMA_MODEL_NAME,
    "dam": SMLMA_MODEL_NAME,
    "sm8": SMLMA_MODEL_NAME
}

CROP_SIZE = config["processor"]["crop_size"]
SIZE_COUNT = config["processor"]["size_count"]
SIZE_THRESHOLD = config["processor"]["size_threshold"]
DETECTION_THRESHOLD = config["processor"]["detection_threshold"]
DISTANCE_THRESHOLD = config["processor"]["distance_threshold"]

MODEL_BASE_PATH = config["paths"]["model"]
PC_NAME = config["pc_name"]

FIRST_GPU_FRACTION = config["gpu"]["fractions"]["first"]
SECOND_GPU_FRACTION = config["gpu"]["fractions"]["second"]

PAD_MODEL_PATH = os.path.join(MODEL_BASE_PATH, PAD_MODEL_NAME, model_filename)
SMCU_MODEL_PATH = os.path.join(MODEL_BASE_PATH, SMCU_MODEL_NAME, model_filename)
CIRCUIT_MODEL_PATH = os.path.join(MODEL_BASE_PATH, CIRCUIT_MODEL_NAME, model_filename)
SMLMA_MODEL_PATH = os.path.join(MODEL_BASE_PATH, SMLMA_MODEL_NAME, model_filename)

pad_model = DetectorClass(PAD_MODEL_PATH, memory_fraction=FIRST_GPU_FRACTION)
smcu_model = DetectorClass(SMCU_MODEL_PATH, memory_fraction=FIRST_GPU_FRACTION)
# elif os.environ.get("MODEL") == "circuit":
#     model = DetectorClass(CIRCUIT_MODEL_PATH, memory_fraction=FIRST_GPU_FRACTION)
# elif os.environ.get("MODEL") == "smlma":
#     model = DetectorClass(SMLMA_MODEL_PATH, memory_fraction=FIRST_GPU_FRACTION)
# elif os.environ.get("MODEL") == "pad_2":
#     model = DetectorClass(PAD_MODEL_PATH, memory_fraction=SECOND_GPU_FRACTION)
# elif os.environ.get("MODEL") == "smcu_2":
#     model = DetectorClass(SMCU_MODEL_PATH, memory_fraction=SECOND_GPU_FRACTION)
# elif os.environ.get("MODEL") == "circuit_2":
#     model = DetectorClass(CIRCUIT_MODEL_PATH, memory_fraction=SECOND_GPU_FRACTION)
# elif os.environ.get("MODEL") == "smlma_2":
#     model = DetectorClass(SMLMA_MODEL_PATH, memory_fraction=SECOND_GPU_FRACTION)

def save_image(image):
    imagename = str(uuid4()) + ".jpg"
    save_path = os.path.join(UPLOAD_PATH, imagename)
    cv2.imwrite(save_path, image)
    return imagename

from util import LOGGER
logger = LOGGER(f"processor-{str(PC_NAME).upper()}", log_path="logs").logger

def processor_run(dataset_id, def_id, ngt_filename, master_path, defect_enc, master_enc, model_type, def_size):
    logger.info(f"Processor running for--{master_path}--{ngt_filename}--{model_type}")
    while True:
        counter = 0
        check = False
        try:
            defect_image = decode(defect_enc)
            master_image = decode(master_enc)
            if model_type == "smcu":
                check = preprocess_orange(defect_image, master_image)
                if check:
                    MODEL_NAME = ORANGE_DETECTION
                    logger.info(f"{datetime.now()} SMCU - ORANGE - {check}")
                if not check:
                    if int(def_size) > BIG_SIZE_THRES:
                        check, save_im = preprocess_big(defect_image, master_image)
                    if check:
                        logger.info(f"{datetime.now()} SMCU - BIG - {check}")
                if not check:
                    if int(def_size) < WHITE_SIZE_THRES:
                        check, save_im = preprocess_white(defect_image, master_image)
                    if check:
                        MODEL_NAME = WHITE_DETECTION
                        logger.info(f"{datetime.now()} SMCU - WHITE - {check}")
                if not check:
                    image = crop_image(defect_image, CROP_SIZE)
                    check, save_im = smcu_model.detect(image, threshold=DETECTION_THRESHOLD, dist_threshold=DISTANCE_THRESHOLD, pixel_threshold=SIZE_THRESHOLD, size_count=SIZE_COUNT )
                    logger.info(f"{datetime.now()} SMCU - DNN - {check}")
                    MODEL_NAME = SMCU_MODEL_NAME
            elif model_type == "pad" or model_type == "qfp":
                if int(def_size) > BIG_SIZE_THRES:
                        check, save_im = preprocess_big(defect_image, master_image)
                if check:
                    logger.info(f"{datetime.now()} PAD - BIG - {check}")
                    MODEL_NAME = BIG_DETECTION
                if not check:
                    image = crop_image(defect_image, CROP_SIZE)
                    check, save_im = pad_model.detect(image, threshold=DETECTION_THRESHOLD, dist_threshold=DISTANCE_THRESHOLD, pixel_threshold=SIZE_THRESHOLD, size_count=SIZE_COUNT )
                    logger.info(f"{datetime.now()} PAD - DNN - {check}")
                    MODEL_NAME = PAD_MODEL_NAME
            # else:
            #     defect_image = crop_image(defect_image, CROP_SIZE)
            #     check, score = model.detect(defect_image, threshold=DETECTION_THRESHOLD, dist_threshold=DISTANCE_THRESHOLD)
            #     logger.info(f"{model_type.upper()} - DNN - {check}")
            image_name = save_image(save_im)
            update_defectPoint(dataset_id, def_id, check, MODEL_NAME, image_name)
            break
        except Exception as inst:
            traceback.print_exc()
            logger.critical(f"{inst}")
            if counter == 3:
                break
            counter += 1