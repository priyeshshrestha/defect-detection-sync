import argparse
import os
import argparse
from pyaml_env import parse_config
from cropper import cropper_run
from db_request import create_lot, update_lot

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'config.yaml')
config = parse_config(config_path)

DIR_PATH = config["paths"]["root"]
VUE_APP_API_URL = config["vue"]

def run_process(date, lot, machine):
    dataset_id = create_lot(lot, date, os.path.join(DIR_PATH, machine, date, part, lot))
    part= lot.split("_")[0]
    lots = os.listdir(os.path.join(DIR_PATH, machine, date, part))
    postfix = ""
    if "curdata_st2" in lots:
        postfix = "_B"
    elif "curdata_st4" in lots:
        postfix = "_B_2"
    for side in os.listdir(os.path.join(DIR_PATH, machine, date, part, lot)):
        if not os.path.isfile(os.path.join(DIR_PATH, machine, date, part, lot, side)):
            if side.endswith(postfix):
                NGT_PATH = os.path.join(DIR_PATH, machine, date, part, lot, side, "ngpointdata")
                # START - NGPOINTDATA EXISTS
                if os.path.exists(NGT_PATH):
                    files = sorted(os.listdir(NGT_PATH), key=lambda x: int(x.split(".")[0].split("_")[-1]))
                    # START - FILE LOOP
                    for filename in files:
                        FILE_PATH = os.path.join(NGT_PATH, filename)
                        cropper_run(FILE_PATH, dataset_id)
                    update_lot(dataset_id)
                        
                        
    

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--machine', metavar="", dest='machine', default=None, required=False, type=str)
    parser.add_argument('-l', '--lot', metavar="", dest='lot', default=None, required=False, type=str)
    parser.add_argument('-d', '--date', metavar="", dest='date', default=None, required=False, type=str)
    known_args, _ = parser.parse_known_args(argv)
    run_process(known_args.date, known_args.lot, known_args.machine)
    

if __name__ == '__main__':
    main()