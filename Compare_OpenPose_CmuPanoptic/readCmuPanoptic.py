import numpy as np
import json
import os

from keyPointClass import keyPointClass

import panutils

def readPanopticJson(panopPath, seq_name, panel, node):
    hd_skel_json_path = panopPath+'/hdPose3d_stage1_coco19/'

    keyPoints = []

    with open(panopPath + '\\calibration_{0}.json'.format(seq_name)) as cfile:
        calib = json.load(cfile)

    cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}

    for k, cam in cameras.items():
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3, 1))

    for skel_json_fname in os.listdir(hd_skel_json_path):
        frameNo = int(skel_json_fname[12:20])
        with open(hd_skel_json_path+skel_json_fname) as dfile:
            bframe = json.load(dfile)

        cam = cameras[(panel, node)]
        
        for pNo, body in enumerate(bframe['bodies']):
            # There are 19 3D joints, stored as an array [x1,y1,z1,c1,x2,y2,z2,c2,...]
            # where c1 ... c19 are per-joint detection confidences
            skel = np.array(body['joints19']).reshape((-1, 4)).transpose()

            # Project skeleton into view (this is like cv2.projectPoints)
            pt = panutils.projectPoints(skel[0:3, :],
                                        cam['K'], cam['R'], cam['t'],
                                        cam['distCoef'])

            # Show only points detected with confidence
            valid = skel[3, :] > 0.1

            pt = pt[:,valid]

            #print(pt)
            
            kp = keyPointClass(pNo, pt, panel, node, frameNo)

        keyPoints.append(kp)

    return keyPoints
        


if __name__=='__main__':
    filepath = 'E:\Dersler\Master\Computer Vision\Term Project\CompareOpenPoseCmu\\sample'
    sequence = 'sample'
    readPanopticJson(filepath, sequence, 0, 0)
