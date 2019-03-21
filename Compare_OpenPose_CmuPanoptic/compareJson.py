import numpy as np
import os
from math import pow
from math import sqrt

from readCmuPanoptic import readPanopticJson
from readOpenPose import readOpenPoseJson


def compareJson(panopJson, opJson, frame):
    # print(panopJson[0].__dict__)
    # print(opJson[0].__dict__)
    try:
        op_person = opJson[0]
    except:
        print("OpenPose returned none at:" + str(panopJson[0].node))
        return np.zeros(15)

    # OpenPose to CMU Panoptic Joint Correspondance List
    # joint_crsp[2] = 9 means:
    # 2nd joint on OpenPose is the 9th joint on CMU Panoptic annotation.
    joint_crsp = [1, 0, 9, 10, 11, 3, 4, 5, 2, 12, 13, 14, 6, 7, 8]

    # Create empty array to store joint errors
    err = np.zeros(15)

    panop_person = panopJson[frame]
    panop_kp = panop_person.keypoints

    minErr = float('Inf')

    for op_person in opJson:
        opKp = op_person.keypoints

        # j = joint number
        # p = panoptic joint number
        for j in range(15):
            if opKp[0, j] != 0:
                p = joint_crsp[j]
                err[j] = err[j] + sqrt(pow(opKp[0, j] - panop_kp[0, p], 2) + pow(opKp[1, j] - panop_kp[1, p], 2))

        # Take mean of joints 1,2,5,8,9,12 and check
        #  if it's minimum across all persons
        err_sum = err[1]+err[2]+err[5]+err[8]+err[9]+err[12]
        err_mean = np.mean(err_sum)

        '''
        ----DEBUG----
        print("Person: "+str(op_person.personNo))
        print("Frame: "+str(frame))
        print("Sum of error: " + str(err_sum))
        print("Min Err: " + str(minErr))
        print(err)
        ----DEBUG----
        '''

        if minErr > err_mean:
            minErr = err_mean
            result = err
            # if it's the minimum error, save it.
        err = np.zeros(15)

    return result


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    filepath = 'E:\Dersler\Master\Computer Vision\Term Project\CompareOpenPoseCmu\\sample'
    sequence = 'sample'

    cwd = os.getcwd()
    errorFilePath = os.path.join(cwd, "Errors")
    create_dir(errorFilePath)

    totalErr = np.zeros(15)

    for camNo in range(31):
        panopJson = readPanopticJson(filepath, sequence, 0, camNo)
        camFilePath = os.path.join(errorFilePath, "%02d" % camNo)
        create_dir(camFilePath)
        if camNo==26:
            idk = panopJson
        for frame in range(101):
            finalPath = os.path.join(camFilePath, "%08d" % frame + ".npy")
            opJson = readOpenPoseJson(filepath, camNo, frame)
            error = compareJson(panopJson, opJson, frame)
            np.save(finalPath, error)
