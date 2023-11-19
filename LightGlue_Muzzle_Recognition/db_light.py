import os
import pymongo
from lightglue import SuperPoint
from lightglue.utils import load_image
from bson.binary import Binary
import pickle



client = pymongo.MongoClient("mongodb://localhost:27017") 
db = client["glue_db"]  
collection = db["light"] 

extractor = SuperPoint(max_num_keypoints=512).eval().cpu()
image_paths = ['A.jpg', 'B.jpg', 'C.jpg']

for img_path in image_paths:
    image = load_image(img_path).cpu()
    feats = extractor.extract(image)
    features_binary = pickle.dumps(feats['descriptors'])
    data_to_insert = {
        "image_path": img_path,
        "features": features_binary,
    }
    collection.insert_one(data_to_insert)

client.close()
