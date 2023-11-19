from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
import os


extractor = SuperPoint(max_num_keypoints=512).eval().cpu()  # load the extractor
matcher = LightGlue(features='superpoint', depth_confidence=0.9, width_confidence=0.95, n_layers=7).eval().cpu()



image0 = load_image('not_match_81_BRIGHT_334964.jpg').cpu()
path = r"D:\Light_Glue_Full\TEST\main_server\animals"
for i in os.listdir(path):

    
    feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None

    

    img = os.path.join(path,i)
    image1 = load_image(img).cpu()




    # extract local features
    feats1 = extractor.extract(image1)

    # match the features

    
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    print("No of Matches: ",len(matches), img)
