import os
import json
import requests
import traceback
from pyaml_env import parse_config

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'config.yaml')
config = parse_config(config_path)

DIR_PATH = config["paths"]["root"]
VUE_APP_API_URL = config["vue"]

def create_lot(name, date, path):
    try:
        headers = { 'content-type': 'application/json' }
        data = {
            "name": name,
            "date": date,
            "path": path
        }
        res = requests.post(os.path.join(VUE_APP_API_URL, "/v1/datasets"), data=json.dumps(data), headers=headers, timeout=10)
        if res.status_code != 200:
            print("Failed to update status.")
        else:
            response_data = res.json()
        if "data" in response_data and "id" in response_data["data"]:
            return response_data["data"]["id"]
        else:
            raise ValueError("ID not found in response")
    except Exception as e:
        print("Unable to update KCE Socket Dashboard")
        traceback.print_exc()

def update_lot(data_id):
    try:
        headers = { 'content-type': 'application/json' }
        data = {
            "status": "ml_complete"
        }
        res = requests.patch(os.path.join(VUE_APP_API_URL, "/v1/datasets/", data_id), data=json.dumps(data), headers=headers, timeout=10)
        if res.status_code != 200:
            print("Failed to update status.")
    except Exception as e:
        print("Unable to update KCE Socket Dashboard")
        traceback.print_exc()

def create_defectPoint(data_id, side, filename, defectNo, X, Y, X1, Y1, X2, Y2, area, size, defectType):
    try:
        headers = { 'content-type': 'application/json' }
        data = {
                "side": side,
                "filename": filename,
                "defectNo": defectNo,
                "X": X,
                "Y": Y,
                "X1": X1,
                "Y1": Y1,
                "X2": X2,
                "Y2": Y2,
                "area": area,
                "size": size,
                "defectType": defectType,
                "model": "",
                "imagePath": ""
            }
        res = requests.post(os.path.join(VUE_APP_API_URL, "/v1/datasets/", data_id, "/defectpoints"), data=json.dumps(data), headers=headers, timeout=10)
        if res.status_code != 200:
            print("Failed to update status.")
        else:
            response_data = res.json()
            if "data" in response_data and "id" in response_data["data"]:
                return response_data["data"]["id"]
            else:
                raise ValueError("ID not found in response")
    except Exception as e:
        print("Unable to update KCE Socket Dashboard")
        traceback.print_exc()

def update_defectPoint(data_id, def_id, prediction, model, imagepath):
    try:
        headers = { 'content-type': 'application/json' }
        data = {
            "ml_defect": prediction,
            "model": model,
            "imagePath": imagepath
        }
        res = requests.patch(os.path.join(VUE_APP_API_URL, "/v1/datasets/", data_id, "/defectpoints/", def_id), data=json.dumps(data), headers=headers, timeout=10)
        if res.status_code != 200:
            print("Failed to update status.")
    except Exception as e:
        print("Unable to update KCE Socket Dashboard")
        traceback.print_exc()