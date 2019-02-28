import numpy as np
import os
from math import pow
from math import sqrt

from readCmuPanoptic import readPanopticJson
from readOpenPose import readOpenPoseJson

def compareJson(panopJson, opJson):

    #print(panopJson[0].__dict__)
    #print(opJson[0][0].__dict__)
    try:
        opPerson = opJson[0][0]
    except:
        print("Encountered none opPerson at:"+str(panopJson[0].node))
        return np.zeros(15)

    jointCorrespondance = [1, 0, 9, 10, 11, 3, 4, 5, 2, 12, 13, 14, 6, 7, 8]
    
    err = np.zeros(15)
    
    for frame in range(101):

        try:
            panopPerson = panopJson[frame]
            opPerson = opJson[0][frame]
        except:
            print(frame)
            print(opJson)
        
        if(panopPerson.personNo == opPerson.personNo):
            panopKp = panopPerson.keypoints
            opKp = opPerson.keypoints
            '''
            print(panopPerson.frame)
            print(opPerson.frame)
            print(panopPerson.node)
            print(opPerson.node)
            '''
            #print(opKp[0,0])
            #print(panopKp[0,1])

            #j = joint number
            #p = panoptic joint number
            for j in range(15):
                if(opKp[0,j] != 0):
                    p = jointCorrespondance[j]
                    err[j] = err[j]+sqrt(pow(opKp[0,j]-panopKp[0,p],2)+pow(opKp[1,j]-panopKp[1,p],2))

    result = np.divide(err,101)                            

    return result
    

if __name__=='__main__':
    filepath = 'E:\Dersler\Master\Computer Vision\Term Project\CompareOpenPoseCmu\\sample'
    sequence = 'sample'

    totalErr = np.zeros(15)
    
    for camNo in range(31):
        panopJson = readPanopticJson(filepath,sequence,0,camNo)
        opJson = readOpenPoseJson(filepath, camNo)
        error = compareJson(panopJson, opJson)
        totalErr = totalErr + error

    print(np.divide(totalErr, 31))
    
