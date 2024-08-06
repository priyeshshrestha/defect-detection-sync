import os
import traceback
import numpy as np
import cv2
from uuid import uuid4
from datetime import datetime, timedelta
from celery.signals import task_failure
from util import encode, color_merge, contains_letters_in_order, contains_specific_numbers, save_error_extention, WrongImageFormat
from processor import processor_run
from pyaml_env import parse_config
from db_request import create_defectPoint
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'config.yaml')
config = parse_config(config_path)

CAM2 = {"A": 1, "B": 2}
CAM4 = {"A1": 1, "A2": 2, "B1": 3, "B2": 4}
FOUR_SIDES = ["_A_1", "_A_2", "_B_1", "_B_2"]
TWO_SIDES = ["_A", "_B"]
DEFECT_TYPES = config["defect_types"]
DEFECT_DICT = {}
AREA_DICT = {
    ("1", "3", "4", "5", "6", "7", "8"): "PAD",
    "2": "QFP",
    "9": "SMCU",
    "10": "SMLMA",
    "11": "CIRCUIT",
    ("12", "13", "19", "20", "21", "22", "23", "24"): "HOLE",
    "14": "SM7",
    "15": "DAM",
    "16": "SM8",
    ("17", "18"): "LEGEND"
}
EXCLUDE_PARTS = ['3T111M0', '1431', '1AL293U04245', '1AL391D02573', '1AL391D02574', '1AL391D02775', '1AL515M04016', '1AL523D02012', '1AL523D02064', '1AL523D02066', '1AL565M04018', '1AS535M04174', '1AT173M06370', '1AT173M06416', '1AT441D02016', '1AT441D02021', '1AT512D02010', '1AT530M04262', '1AT561M04007', '1AT561M04009', '1AT561M04058', '1AT561M04071', '1AT561M04072', '1AT561M04073', '1AT561M04076', '1IL131M06610', '1IL488D02145', '1IL488D02332', '1IL488M04213', '1IL488M04344', '1IL488M06319', '1IL598M04040', '1IL598M04053', '1IT598M08075', '1SS654D02014', '1UT431E08086', '1UT530D02441', '1YT512D02017', '1YT523D02247',
                 '1YT530A04313', '1YT552M04019', '1YT561D02051', '1YX431M08072', '1AT561M04009', '1AT561M04007', '1YT561M04058', '2AL915D02261', '2AL915D02448', '2AS915D02446', '2AS915D02453', '2AS915M04336', '2AT841D02149', '2AT943D02005', '2YT841M06137', '3YE382D02106', '3YE382M04105', '3YT262M04006', '3YT382M04060', '3YT382M06094', '3AT981M0617', '3AT981M06193', '4AE510M04269', '4AH501R04196', '4AT565D02227', '4AT565M04312', '4AT566D02217', '4AT566M04235', '4AT575M06163', '4AT620M04048', '4IL643M06013', '4IT590M04049', '4IT632M04047', '4IT632M04050', '4UG595Y06336', '4US620M04161', '4UT620Z10213', '4UT620Z10240', 'RC']
PC_NAME = config["pc_name"]
DIR_PATH = config["paths"]["root"]
CROP_SIZE = config["cropper"]["crop_size"]
CROPPER_SAVE_PATH = config["paths"]["cropper"]

wrong_format_parts = []

from util import LOGGER
logger = LOGGER(f'cropper-{str(config["pc_name"]).upper()}', log_path="logs").logger


def process_line(line, idx):
    coords = line[2 + idx]
    words = coords.split(",")
    defect_no = words[0]
    X = words[1]
    Y = words[2]
    area = int(words[3])
    if area < 25:
        defect_type = str(
            next(v for k, v in AREA_DICT.items() if str(area) in k)).lower()
    else:
        defect_type = "others"
    size = words[4]
    x1 = int(words[5])
    x2 = int(words[6])
    y1 = int(words[7])
    y2 = int(words[8])
    function = words[9]
    string_name = str(uuid4()) + "_" + str(defect_no) + \
        "_" + str(X) + "_" + str(Y) + ".jpg"
    data = {
        "defect_no": defect_no,
        "X": X,
        "Y": Y,
        "area": area,
        "size": size,
        "X1": x1,
        "Y1": y1,
        "X2": x2,
        "Y2": y2,
        "function": function,
        "var": -1,
    }
    return X, Y, string_name, defect_type, data


def crop_image(image, x, y, width, height):
    point_with_padding = (int(x) - int(CROP_SIZE / 2), int(y) - int(CROP_SIZE / 2), int(x) + int(CROP_SIZE / 2), int(y) + int(CROP_SIZE / 2))
    output = np.zeros((CROP_SIZE, CROP_SIZE, 3))
    (x1, y1, x2, y2) = point_with_padding
    if x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
        cp_img = image[y1:y2, x1:x2]
        output[0:CROP_SIZE, 0:CROP_SIZE] = cp_img
    else:
        if x1 < 0:
            if y1 < 0:
                cp_img = image[0:y2, 0:x2]
                hh, ww, _ = cp_img.shape
                output[CROP_SIZE-hh:CROP_SIZE, CROP_SIZE-ww:CROP_SIZE] = cp_img
            elif y2 > height:
                cp_img = image[y1:height, 0:x2]
                hh, ww, _ = cp_img.shape
                output[0:hh, CROP_SIZE-ww:CROP_SIZE] = cp_img
            else:
                cp_img = image[y1:y2, 0:x2]
                hh, ww, _ = cp_img.shape
                output[CROP_SIZE-hh:CROP_SIZE, CROP_SIZE-ww:CROP_SIZE] = cp_img
        elif x2 > width:
            if y1 < 0:
                cp_img = image[0:y2, x1:width]
                hh, ww, _ = cp_img.shape
                output[CROP_SIZE-hh:CROP_SIZE, 0:ww] = cp_img
            elif y2 > height:
                cp_img = image[y1:height, x1:width]
                hh, ww, _ = cp_img.shape
                output[0:hh, 0:ww] = cp_img
            else:
                cp_img = image[y1:y2, x1:width]
                hh, ww, _ = cp_img.shape
                output[CROP_SIZE-hh:CROP_SIZE, 0:ww] = cp_img
        else:
            if y1 < 0:
                cp_img = image[0:y2, x1:x2]
                hh, ww, _ = cp_img.shape
                output[CROP_SIZE-hh:CROP_SIZE, 0:ww] = cp_img
            else:
                cp_img = image[y1:height, x1:x2]
                hh, ww, _ = cp_img.shape
                output[0:hh, 0:ww] = cp_img
    return output


def process_defect(args):
    dataset_id, lines, side, part, i, img_h, img_w, image_colored, master_colored, ngt_uuid, ngt_upload_time, cropper_start_time, machine, date, lot, ngt_filename, ngt_path, defect_count, total_record = args
    x, y, string_name, defect_type, data = process_line(lines[side]["line"], i)
    check = any(item in part.split("-") for item in EXCLUDE_PARTS)
    if not check:
        check = any(item in part.split(" ") for item in EXCLUDE_PARTS)
    if not check:
        check = contains_letters_in_order(part.split("-")[0], EXCLUDE_PARTS[0])
    if not check:
        check = contains_specific_numbers(part.split("-")[0], EXCLUDE_PARTS[1])
    if not check:
        if defect_type in DEFECT_TYPES:
            DEFECT_DICT[str(defect_type)] += 1
            cropped_image = crop_image(image_colored.copy(), x, y, img_w, img_h)
            cropped_master = crop_image(master_colored.copy(), x, y, img_w, img_h)
            cv2.imwrite("./logs/a.jpg",cropped_image)
            def_enc = encode(cropped_image)
            mas_enc = encode(cropped_master)
            logger.info(f"Sending {data['defect_image']} for processing.")
            def_id = create_defectPoint(side, ngt_filename, data["defect_no"], data["X"], data["Y"], data["X1"], data["Y1"], data["X2"], data["Y2"], data["area"], data["size"], defect_type)
            processor_run(dataset_id, def_id, data["ngt_filename"], data["master_image"], def_enc, mas_enc, data["defect_type"], data["size"])


def crop_sides(dataset_id, ngt_file_path, side_path, lines, ngt_uuid, total_record, master_img_folder, cam_side):
    directories = ngt_file_path.split("/")
    machine = directories[- 7]
    date = directories[- 6]
    part = directories[- 5]
    lot = directories[- 4]
    side = (directories[len(directories) - 3].split("_"))[-1]
    ngt_filename = os.path.split(ngt_file_path)[1]
    base_name = ngt_filename.split(".")[0]
    cropper_start_time = datetime.now()
    for i in DEFECT_TYPES:
        DEFECT_DICT[i] = 0
    side = side_path.split("/")[-1]
    side_no = side.split("_")[-1]
    if side_no.isnumeric():
        side_no = "".join(side.split("_")[-2:])
        cam_side = str(CAM4[side_no])
    else:
        cam_side = str(CAM2[side_no])
    ngt_path = os.path.join("/", side_path, "ngpointdata", ngt_filename)
    ngt_upload_time = datetime.fromtimestamp(os.path.getmtime(ngt_path)).strftime("%Y-%m-%d %H:%M:%S")
    if len(lines[side]["line"]) > 0:
        defect_count = int(lines[side]["line"][1])
    else:
        defect_count = 0
    data = {}
    if "ERR" in base_name or "OK" in base_name or defect_count == 0: # or defect_count >= 1000:\
        logger.info(f"ERR/OK file")
    else:
        logger.info(f"{date}/{side}/{base_name}")
        image_path = directories[:len(directories) - 3]
        image_path = os.path.join("/", *image_path, side, "checkimg")
        image_blue = os.path.join(image_path, "B_" + base_name + ".jpg")
        image_red = os.path.join(image_path, "R_" + base_name + ".jpg")
        image_green = os.path.join(image_path, "G_" + base_name + ".jpg")
        if not os.path.exists(image_blue) or not os.path.exists(image_red) or not os.path.exists(image_green):
            raise WrongImageFormat({part: "Image does not exist. Possible extension mismatch."})
        image_colored = color_merge(image_blue, image_green, image_red)
        img_h, img_w, _ = image_colored.shape
        master_image_path = os.path.join(CROPPER_SAVE_PATH, date, part, lot, side, "master.jpg")
        if os.path.exists(master_image_path):
            master_colored = cv2.imread(master_image_path)
        else:
            master_path = directories[:len(directories) - 4]
            master_path = os.path.join("/", *master_path, master_img_folder, "camera" + cam_side)
            master_blue = os.path.join(master_path, "KIBANCUR_B.jpg")
            master_red = os.path.join(master_path, "KIBANCUR_R.jpg")
            master_green = os.path.join(master_path, "KIBANCUR_G.jpg")
            if not os.path.exists(master_blue) or not os.path.exists(master_red) or not os.path.exists(master_green):
                raise WrongImageFormat({part: "Image does not exist. Possible extension mismatch."})
            master_colored = color_merge(master_blue, master_green, master_red)
        for i in range(0, defect_count):
            process_defect((dataset_id, lines, side, part, i, img_h, img_w, image_colored, master_colored, ngt_uuid, ngt_upload_time, cropper_start_time, machine, date, lot,
                      ngt_filename, ngt_path, defect_count, total_record))


def crop( ngt_file_path, dataset_id):
    directories = ngt_file_path.split("/")
    lot = directories[- 4]
    side = (directories[len(directories) - 3].split("_"))[-1]
    ngt_filename = os.path.split(ngt_file_path)[1]
    side_path_list = []
    if side.isnumeric():
        side = "".join(directories[len(directories) - 3].split("_")[-2:])
        master_img_folder = "curdata_st4"
        cam_side = str(CAM4[side])
        for side_suffix in FOUR_SIDES:
            side_path = os.path.join(*directories[:len(directories) - 3], str(lot) + str(side_suffix))
            side_path_list.append(side_path)
    else:
        master_img_folder = "curdata_st2"
        cam_side = str(CAM2[side])
        for side_suffix in TWO_SIDES:
            side_path = os.path.join(*directories[:len(directories) - 3], str(lot) + str(side_suffix))
            side_path_list.append(side_path)
    ngt_uuid = str(uuid4())
    total_record = 0
    lines = {}
    for side_path in sorted(side_path_list):
        side = side_path.split("/")[-1]
        ngt_path = os.path.join("/", side_path, "ngpointdata", ngt_filename)
        lines[side] = {}
        f = open(os.path.join(ngt_path), "r")
        line = f.readlines()
        lines[side]["line"] = line
        if len(lines[side]["line"]) > 0:
            if int(lines[side]["line"][1]) == 0:
                total_record += 1
            else:
                total_record += int(lines[side]["line"][1])
        else:
            total_record += 1
    for side_path in sorted(side_path_list):
        crop_sides( dataset_id, ngt_file_path, side_path, lines, ngt_uuid, total_record, master_img_folder, cam_side)


def cropper_run(ngt_file_path, dataset_id):
    print(f"Cropper running for file: {ngt_file_path}")
    logger.info(f"Cropper running for file: {ngt_file_path}")
    # ngt_filename = ngt_file_path.split("/")[-1]
    try:
        crop(ngt_file_path, dataset_id)
        logger.info(f"Cropping complete for file: {ngt_file_path}")
    except Exception as e:
        part, message = e.args
        traceback.print_exc()
        logger.critical(f"{message}")
        if part not in wrong_format_parts:
            wrong_format_parts.append(part)
            msg = f"ERROR in {PC_NAME} part {part} : {message}"
            logger.critical(msg)
