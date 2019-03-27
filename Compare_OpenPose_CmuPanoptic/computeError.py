import numpy as np
import os

cwd = os.getcwd()

def calcTotal(seq):

    errorFP = os.path.join(cwd,"Errors"+seq)

    for camNo in range(31):

        nextFP = os.path.join(errorFP, "%02d" % camNo)
        savefile = os.path.join(nextFP, "total.npy")
        if os.path.exists(savefile):
            continue
        err = np.zeros(15)
        countList = np.zeros(15)
        if not os.path.exists(nextFP):
            continue
        for file in os.listdir(nextFP):
            if 'total.npy' in file:
                continue
            errorFile = os.path.join(nextFP, file)
            currentErr = np.load(errorFile)
            countList = np.where(currentErr == 0, countList, countList+1)
            err = err + currentErr

        print(countList)
        print(err)
        meanErr = np.divide(err, countList)
        meanErr = np.nan_to_num(meanErr)
        print(meanErr)
        np.save(savefile, meanErr)
        print(camNo," completed")
        '''
        for frame in range(101):

            frameNo = "%08d" % frame
            errorFile = os.path.join(nextFP, frameNo+".npy")

            err = err + np.load(errorFile)


        np.save(savefile,err)
        '''
    return 0


def printErr(camNo, seq):

    errorFP = os.path.join(cwd,"Errors"+seq)
    nextFP = os.path.join(errorFP, "%02d" % camNo)
    loadfile = os.path.join(nextFP,"total.npy")

    error = np.load(loadfile)

    maxValue = np.amax(error)
    maxInd = np.where(error == np.amax(error))
    
    #print("Cam no: "+ str(camNo) + ". Maximum Error: "+str(maxValue)+" at joint: "+str(maxInd[0]))
    #print(str(camNo) + ","+str(maxValue)+","+str(maxInd[0]))
    sumErr = np.sum(error)

    print(str(camNo)+ "," +str(maxValue) + "," + str(maxInd[0]) + "," + str(sumErr))
    
    return error,maxValue,maxInd


def calcMaxErr(seq):

    jointMaxErr = np.zeros(15)
    jointErrCount = np.zeros(15)
    
    totalErr = np.zeros(15)
    for i in range(31):
        if '171204_pose3' in seq and (i == 20 or i == 21):
            continue
        error, maxV, maxInd = printErr(i,seq)
        jointMaxErr[maxInd] += maxV
        jointErrCount[maxInd] += 1
        totalErr = np.add(totalErr, error)

    if '171204_pose3' in seq:
        totalErr = np.divide(totalErr, 29)
    else:
        totalErr = np.divide(totalErr, 31)

    avgErr = np.nan_to_num(np.divide(jointMaxErr, jointErrCount))
    

    maxErrJoint = np.where(jointMaxErr == np.amax(jointMaxErr))
    maxErr = np.amax(jointMaxErr)
    print("Maximum Error Joint: " + str(maxErrJoint[0]) + " at error: " + str(maxErr))
    print("Average error of " + str(maxErrJoint[0]) + " is: " + str(avgErr[maxErrJoint[0]]))

    avgMaxErr = np.amax(avgErr)
    avgMaxErrJoint = np.where(avgErr == np.amax(avgErr))

    print("Max. Avg. Err. Joint: " + str(avgMaxErrJoint[0]) + " at error: " + str(avgMaxErr))

    print("Total Error: ")
    print(totalErr)
    
    return 0


if __name__=='__main__':
    seq = "171204_pose3"
    calcTotal(seq)
    calcMaxErr(seq)
