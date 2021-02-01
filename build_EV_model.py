import csv
import json



def load_data(site='jpl'):
    simplified_data = []
    source = "ev-data/office1-only3.json"
    if site == 'jpl':
        source = 'ev-data/JPL-1-9-18--4-6-19.json'
    elif site == 'caltech':
        source = 'ev-data/caltech-1-4-18--4-6-19.json'
    elif site == 'office1':
        source = "ev-data/office1-25-3-18--4-6-19.json"
    with open(source) as json_file:
        data = json.load(json_file)
        for d in  data["_items"]:
            simplified_data.append({"connectionTime": d["connectionTime"],
                                    "disconnectTime":d["disconnectTime"],
                                    "doneChargingTime":d["doneChargingTime"],
                                    "kWhDelivered":d["kWhDelivered"]})
    return simplified_data



