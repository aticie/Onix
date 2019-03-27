import numpy as np
import os
from math import pow
from math import sqrt
import time

from readCmuPanoptic import readPanopticJson
from readOpenPose import readOpenPoseJson


def compareJson(panop_person, opJson):

    # print(panopJson[0].__dict__)
    # print(opJson[0].__dict__)
    try:
        op_person = opJson[0]
    except:
        print("OpenPose returned none at frame:" + str(panop_person.frame))
        return np.zeros(15)

    # OpenPose to CMU Panoptic Joint Correspondence List
    # joint_crsp[2] = 9 means:
    # 2nd joint on OpenPose is the 9th joint on CMU Panoptic annotation.
    joint_crsp = [1, 0, 9, 10, 11, 3, 4, 5, 2, 12, 13, 14, 6, 7, 8]

    # Create empty array to store joint errors
    err = np.zeros(15)

    # Panoptic ground truths
    panop_kp = panop_person.keypoints

    minErr = float('Inf')

    for op_person in opJson:
        # OpenPose keypoint guesses
        opKp = op_person.keypoints

        # j = joint number
        # p = panoptic joint number
        for j in range(15):
            if opKp[0, j] != 0:
                p = joint_crsp[j]
                # Apply image resolution constraints
                # If a joint is outside 1920x1080, don't count error
                if not (panop_kp[0, p] > 1920 or panop_kp[0, p] < 0 or panop_kp[1, p] > 1080 or panop_kp[1, p] < 0):
                    err[j] = err[j] + sqrt(pow(opKp[0, j] - panop_kp[0, p], 2) + pow(opKp[1, j] - panop_kp[1, p], 2))

        # Take mean of joints 1,2,5,8,9,12 and check
        # if it's minimum across all persons
        # to make sure we are calculating error on the correct person
        # Joints 1,2,5,8,9,12 are the upper body joints
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
    filepath = 'E:\Dersler\Master\Computer Vision\Term Project\CompareOpenPoseCmu\\171204_pose3'
    sequence = '171204_pose3'

    cwd = os.getcwd()
    errorFilePath = os.path.join(cwd, "Errors"+sequence)
    create_dir(errorFilePath)

    totalErr = np.zeros(15)

    for camNo in range(22,31):

        panopJson = readPanopticJson(filepath, sequence, 0, camNo)

        camFilePath = os.path.join(errorFilePath, "%02d" % camNo)

        create_dir(camFilePath)

        if camNo == 20 or camNo == 21:
            continue

        for item in panopJson:
            frame = item.frame
            finalPath = os.path.join(camFilePath, "%08d" % frame + ".npy")
            if os.path.exists(finalPath):
                continue
            opJson = readOpenPoseJson(filepath, camNo, frame)
            error = compareJson(item, opJson)
            np.save(finalPath, error)

