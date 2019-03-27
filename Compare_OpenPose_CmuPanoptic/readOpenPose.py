import numpy as np
import json
import os

from keyPointClass import keyPointClass

def readOpenPoseJson(kpPath, camNo, frame):

    node = '%02d' % camNo
    panel = '00'
    if 'sample' in os.path.basename(os.path.normpath(kpPath)):
        frameNo = '%08d' % frame
        jsonName = panel + "_" + node + "_" + frameNo + "_keypoints.json"
        kpPath = kpPath + '\\OpenPose_out'
    else:
        frameNo = '%012d' % frame
        jsonName = "hd_" + panel + "_" + node + "_" + frameNo + "_keypoints.json"
        kpPath = kpPath + '\\OpenPose'

    camNoPath = os.path.join(kpPath, panel + '_' + node)

    # Output from OpenPose must be under OpenPose_out folder


    #Contents of the OpenPose should be as follows:
    
    #OpenPose
    # |
    # ---> 00_00 (camNoFolder)
    #      |
    #       ---> hd_00_00_000000000001_keypoints.json (keypointsJson)
    #       ---> hd_00_00_000000000002_keypoints.json
    #       ...
    # ---> 00_01
    #      |
    #      ...

    # Open the folder with selected camNo

    
    # Create empty keypoint array to kp values
    keyP = []

    op_skel_json = os.path.join(camNoPath, jsonName)
    
    with open(op_skel_json) as dfile:
        # Load OpenPose Json file
        opJson = json.load(dfile)

    for pNo, person in enumerate(opJson['people']):
        keypoints = np.array(person['pose_keypoints_2d']).reshape((-1,3)).transpose()
        kp = keyPointClass(pNo, keypoints, panel, node, frame)
        keyP.append(kp)

    return keyP

if __name__=='__main__':
    filepath = 'E:\Dersler\Master\Computer Vision\Term Project\CompareOpenPoseCmu\\sample'
    sequence = 'sample'
    opKp = readOpenPoseJson(filepath,1)
