import numpy as np
import json
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['image.interpolation'] = 'nearest'

# For camera projection (with distortion)
import panutils

# Setup paths
data_path = '../'
seq_name = '171204_pose1_sample'

hd_skel_json_path = data_path+seq_name+'/hdPose3d_stage1_coco19/'
hd_face_json_path = data_path+seq_name+'/hdFace3d/'
hd_hand_json_path = data_path+seq_name+'/hdHand3d/'
hd_img_path = data_path+seq_name+'/hdImgs/'

colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()

# Load camera calibration parameters
with open(data_path+seq_name+'/calibration_{0}.json'.format(seq_name)) as cfile:
    calib = json.load(cfile)

# Cameras are identified by a tuple of (panel#,node#)
cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

# Convert data into numpy arrays for convenience
for k,cam in cameras.items():
    cam['K'] = np.matrix(cam['K'])
    cam['distCoef'] = np.array(cam['distCoef'])
    cam['R'] = np.matrix(cam['R'])
    cam['t'] = np.array(cam['t']).reshape((3,1))


err = np.zeros(15)

for camNo in range(31):
    cam = cameras[(0,camNo)]
    for frameNo in range(100):
    
        skel_json_fname = hd_skel_json_path+'body3DScene_{0:08d}.json'.format(frameNo)
        op_skel_json = hd_img_path+'{0:02d}_{1:02d}/json/{0:02d}_{1:02d}_{2:08d}_keypoints.json'.format(cam['panel'], cam['node'], frameNo)
        
        with open(skel_json_fname) as dfile:
            bframe = json.load(dfile)
        
        with open(op_skel_json) as dfile:
            opbframe = json.load(dfile)
        
        for body in bframe['bodies']:
        
            # There are 19 3D joints, stored as an array [x1,y1,z1,c1,x2,y2,z2,c2,...]
            # where c1 ... c19 are per-joint detection confidences
            skel = np.array(body['joints19']).reshape((-1,4)).transpose()

            # Project skeleton into view (this is like cv2.projectPoints)
            pt = panutils.projectPoints(skel[0:3,:],
                          cam['K'], cam['R'], cam['t'], 
                          cam['distCoef'])

            # Show only points detected with confidence
            valid = skel[3,:]>0.1

        maxConfidence = 0
        for person in opbframe['people']:
            oppt_temp = np.array(person['pose_keypoints_2d']).reshape((-1,3)).transpose()
            if(oppt_temp[2,0]>maxConfidence):
                oppt = oppt_temp
                maxConfidence = oppt_temp[2,0]

        err[0:2] = err[0:2] + ((oppt[0, 0:2] - pt[0, 0:2])**2+(oppt[1, 0:2] - pt[1, 0:2])**2)**(1/2)
        err[2:5] = err[2:5] + ((oppt[0, 2:5] - pt[0, 9:12]) ** 2 + (oppt[1, 2:5] - pt[1, 9:12]) ** 2) ** (1 / 2)
        err[5:8] = err[5:8] + ((oppt[0, 5:8] - pt[0, 3:6]) ** 2 + (oppt[1, 5:8] - pt[1, 3:6]) ** 2) ** (1 / 2)
        err[8] = err[8] + ((oppt[0,8 ] - pt[0, 2]) ** 2 + (oppt[1, 8] - pt[1, 2]) ** 2) ** (1 / 2)
        err[9:12] = err[9:12] + ((oppt[0, 9:12] - pt[0, 12:15]) ** 2 + (oppt[1, 9:12] - pt[1, 12:15]) ** 2) ** (1 / 2)
        err[12:15] = err[12:15] + ((oppt[0, 12:15] - pt[0, 6:9]) ** 2 + (oppt[1, 12:15] - pt[1, 6:9]) ** 2) ** (1 / 2)




meanErr = np.divide(err,3100)

print(meanErr)
