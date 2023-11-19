# Animal_Test_Recognition
This repository is used for Animal Recognition using the Cropped Muzzle Images

The basic idea/architecture behind this innovation can be explained through the given equation: 

`````````````````````````````````````````````````````````````````````
Cropped Muzzle --> Feature Extraction --> Comparison with All Muzzles
`````````````````````````````````````````````````````````````````````

Any Object Detection Algorithm can be used for detection of the Region of Interest. After selection of the Required ROI the extarcted part can be used for Fine-Grained Feature Extarction.
The Algorithms of LOFtr and LightGlue can be used for feature extraction. There are also dedicated matchers for both of the latest feature extraction algorithms.

The file

````````````
glue_test.py
`````````````

contains the matching Test code for indivudual animal muzzle with the list available.

