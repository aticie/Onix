import numpy as np
import os
from math import pow
from math import sqrt

from readCmuPanoptic import readPanopticJson
from readOpenPose import readOpenPoseJson

def compareJson(panopJson, opJson, frame):

    #print(panopJson[0].__dict__)
    #print(opJson[0].__dict__)
    try:
        opPerson = opJson[0]
    except:
        print("OpenPose returned none at:"+str(panopJson[0].node))
        return np.zeros(15)

    # OpenPose to CMU Panoptic Joint Correspondance List
    # jointCorrespondance[2] = 9 means:
    # 2nd joint on OpenPose is the 9th joint on CMU Panoptic annotation.
    jointCorrespondance = [1, 0, 9, 10, 11, 3, 4, 5, 2, 12, 13, 14, 6, 7, 8]

    # Create empty array to store joint errors
    err = np.zeros(15)

    
    panopPerson = panopJson[frame]
    panopKp = panopPerson.keypoints

    minErr = float('Inf')
    
    for opPerson in opJson:
        opKp = opPerson.keypoints

       
        #j = joint number
        #p = panoptic joint number
        for j in range(15):
            if(opKp[0,j] != 0):
                p = jointCorrespondance[j]
                err[j] = err[j]+sqrt(pow(opKp[0,j]-panopKp[0,p],2)+pow(opKp[1,j]-panopKp[1,p],2))
                
        # Sum all joint errors and check if it's minimum across all persons
        errSum = sum(err)
        '''
        ----DEBUG----
        print("Person: "+str(opPerson.personNo))
        print("Frame: "+str(frame))
        print("Sum of error: " + str(errSum))
        print("Min Err: " + str(minErr))
        print(err)
        ----DEBUG----
        '''
        if minErr > errSum:
            minErr = errSum
            result = err
            # if it's the minimum error, save it.
        err = np.zeros(15)

    return result


def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__=='__main__':
    filepath = 'E:\Dersler\Master\Computer Vision\Term Project\CompareOpenPoseCmu\\sample'
    sequence = 'sample'

    cwd = os.getcwd()
    errorFilePath = os.path.join(cwd,"Errors")
    createDir(errorFilePath)

    totalErr = np.zeros(15)

    for camNo in range(31):
        panopJson = readPanopticJson(filepath,sequence,0,camNo)
        camFilePath = os.path.join(errorFilePath,"%02d" % camNo)
        createDir(camFilePath)
        for frame in range(101):
            finalPath = os.path.join(camFilePath, "%08d" % frame+".npy")
            opJson = readOpenPoseJson(filepath, camNo, frame)
            error = compareJson(panopJson, opJson, frame)
            np.save(finalPath,error)


    
