import numpy as np
import json
import os

from keyPointClass import keyPointClass

def readOpenPoseJson(kpPath, camNo):

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

    opKp = []
    for camNoFolder in os.listdir(kpPath):
        camNoPath = os.path.join(kpPath,camNoFolder)

        panel = int(camNoFolder[0:2])
        node = int(camNoFolder[3:5])

        if(node != camNo):
            continue
        
        keyP = []
        
        for frame, keypointsJson in enumerate(os.listdir(camNoPath)):
            op_skel_json = os.path.join(camNoPath,keypointsJson)
            
            with open(op_skel_json) as dfile:
                # Load OpenPose Json file
                opJson = json.load(dfile)

            for pNo, person in enumerate(opJson['people']):
                keypoints = np.array(person['pose_keypoints_2d']).reshape((-1,3)).transpose()
                kp = keyPointClass(pNo, keypoints, panel, node, frame)
                
            keyP.append(kp)


        opKp.append(keyP)

    return opKp

if __name__=='__main__':
    filepath = 'E:\Dersler\Master\Computer Vision\Term Project\CompareOpenPoseCmu\\sample'
    sequence = 'sample'
    opKp = readOpenPoseJson(filepath)
