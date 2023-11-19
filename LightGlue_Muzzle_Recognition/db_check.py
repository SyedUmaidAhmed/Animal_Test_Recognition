import os
import pymongo
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image, rbd
from bson.binary import Binary
import pickle
import torch

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["glue_db"]
collection = db["light"]

# Initialize SuperPoint extractor and Light Glue matcher
extractor = SuperPoint(max_num_keypoints=512).eval().cpu()
matcher = LightGlue().eval().cpu()

# Load the query image
query_image_path = 'C.jpg'  # Replace with the path to your query image
query_image = load_image(query_image_path).cpu()

# Extract SuperPoint features from the query image
query_feats = extractor.extract(query_image)

# Query MongoDB to retrieve SuperPoint features from other images
cursor = collection.find({}, {"features": 1, "image_path": 1})

best_match = None
best_match_score = 0.0

# Match features
for doc in cursor:
    stored_features = pickle.loads(doc["features"])  # Deserialize the binary features

    # Match features using Light Glue or any other matching algorithm
    stored_features = [torch.tensor(x) for x in stored_features]  # Convert to a list of tensors
    query_feats = [query_feats]  # Convert to a list of tensors
    matches = matcher({"image0": query_feats, "image1": stored_features})
    matches = matches["matches"]

    # Calculate a similarity score, e.g., the number of matches or another scoring metric
    similarity_score = len(matches)

    # Check if this is the best match so far
    if similarity_score > best_match_score:
        best_match_score = similarity_score
        best_match = doc["image_path"]

# Print or store the best match and the matching score
print("Best Match:", best_match, "Score:", best_match_score)

# Close the MongoDB connection
client.close()
