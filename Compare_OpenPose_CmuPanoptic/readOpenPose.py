import numpy as np
import json
import os

from keyPointClass import keyPointClass

def readOpenPoseJson(kpPath, camNo, frame):

    # Output from OpenPose must be under OpenPose_out folder
    kpPath = kpPath+'\\OpenPose_out'

    #Contents of the OpenPose_out should be as follows:
    
    #OpenPose_out
    # |
    # ---> 00_00 (camNoFolder)
    #      |
    #       ---> 00_00_00000001_keypoints.json (keypointsJson
    #       ---> 00_00_00000002_keypoints.json
    #       ...
    # ---> 00_01
    #      |
    #      ...

    # Open the folder with selected camNo
    node = '%02d' % camNo
    frameNo = '%08d' % frame
    panel = '00'
    camNoPath = os.path.join(kpPath,panel+'_'+node)
    jsonName = panel+"_"+node+"_"+frameNo+"_keypoints.json"
    
    # Create empty keypoint array to kp values
    keyP = []

    op_skel_json = os.path.join(camNoPath,jsonName)
    
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
