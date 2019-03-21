import numpy as np
import os

cwd = os.getcwd()

def calcTotal():

    errorFP = os.path.join(cwd,"Errors")

    for camNo in range(31):

        nextFP = os.path.join(errorFP, "%02d" % camNo)
        savefile = os.path.join(nextFP, "total.npy")
        if os.path.exists(savefile):
            continue
        err = np.zeros(15)

        for frame in range(101):

            frameNo = "%08d" % frame
            errorFile = os.path.join(nextFP, frameNo+".npy")

            err = err + np.load(errorFile)


        np.save(savefile,err)

    return 0


def printErr(camNo):

    errorFP = os.path.join(cwd,"Errors")
    nextFP = os.path.join(errorFP, "%02d" % camNo)
    loadfile = os.path.join(nextFP,"total.npy")

    error = np.load(loadfile)

    error = np.divide(error,101)

    maxValue = np.amax(error)
    maxInd = np.where(error == np.amax(error))
    
    #print("Cam no: "+ str(camNo) + ". Maximum Error: "+str(maxValue)+" at joint: "+str(maxInd[0]))
    #print(str(camNo) + ","+str(maxValue)+","+str(maxInd[0]))
    sumErr = np.sum(error)

    print(str(camNo)+ "," + str(sumErr))
    
    return error,maxValue,maxInd


def calcMaxErr():

    err = np.zeros(15)

    jointMaxErr = np.zeros(15)
    jointErrCount = np.zeros(15)
    
    totalErr = np.zeros(15)
    for i in range(31):
        error,maxV,maxInd = printErr(i)
        jointMaxErr[maxInd] += maxV
        jointErrCount[maxInd] += 1
        totalErr = np.add(totalErr,error)

    totalErr = np.divide(totalErr,31)
    err = np.divide(err,31)

    avgErr = np.nan_to_num(np.divide(jointMaxErr,jointErrCount))
    

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

    calcTotal()
    calcMaxErr()
