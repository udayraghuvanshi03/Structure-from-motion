import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

class SFM:
    def __init__(self, array_of_images: list,show_tracked_points:bool,show_scatter_plot:bool):
        self.array_of_images = array_of_images
        self.show_tracked_points=show_tracked_points
        self.show_scatter_plot=show_scatter_plot

    def tracking(self):
        prev_gray=self.array_of_images[0].astype(np.uint8)
        max_pts=800
        quality=0.001
        min_dist=0.01
        prev_corners = cv2.goodFeaturesToTrack(prev_gray,maxCorners=max_pts,qualityLevel=quality,minDistance=min_dist)
        termcrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        feat=[]
        ind=[]
        for i in range(1,len(self.array_of_images)):
            frame_gray=self.array_of_images[i].astype(np.uint8)
            next_corners,status,_=cv2.calcOpticalFlowPyrLK(prev_gray,frame_gray, prev_corners, None, criteria=termcrit)
            good_pts=next_corners[status==1]
            feat.append(next_corners)
            ind.append(status)
            if self.show_tracked_points:
                good_pts_int=np.round(good_pts).astype(np.int32)
                output=self.array_of_images[i].astype(np.uint8)
                color_image = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                # print(status[5])
                for j in range(good_pts_int.shape[0]):
                    cv2.circle(color_image,(good_pts_int[j][0],good_pts_int[j][1]),2,(0,0,255))
                cv2.imshow('Klt tracking',color_image)
                cv2.waitKey(200)
            print(good_pts.shape)
            prev_gray=frame_gray.copy()
            prev_corners = good_pts.reshape(-1,1,2)
        final_feat=[]
        final_good_pts=[]
        for i in range(len(ind)):
            p_temp=feat[i]
            for j in range(i,len(ind)):
                st_temp=ind[j]
                final_good_pts=p_temp[st_temp==1]
                p_temp=final_good_pts.reshape(-1,1,2)
            final_feat.append(final_good_pts)
        # Forming W matrix
        rows=len(final_feat)
        cols=len(final_feat[0])
        W=np.zeros((2*rows,cols))
        for i in range(rows):
            for j in range(cols):
                W[i][j]=final_feat[i][j][0]
                W[i+rows][j]=final_feat[i][j][1]

        # Subtracting the mean
        W_hat=W-W.mean(axis=1).reshape(-1,1)

        return W_hat

    def point_coords(self):
        W_hat=self.tracking()
        u, d, v_transpose = np.linalg.svd(W_hat,full_matrices=True)
        d[3:]=0
        d_f=np.array([[d[0],0,0],[0,d[1],0],[0,0,d[2]]])
        u_f=np.array(u[:,:3])
        v_f=np.array(v_transpose[:3,:])
        d_f=np.sqrt(d_f)
        R_hat=np.dot(u_f,d_f)
        S_hat=np.dot(d_f,v_f)
        # m = number of frames
        m=R_hat.shape[0]//2
        R_i=R_hat[:m,:]
        R_j=R_hat[m:2*m,:]
        G=np.zeros((2*m,6))
        for i in range(m):
            G[2*i,0]=(R_i[i,0]**2)-(R_j[i,0]**2)
            G[2*i+1,0]=R_i[i,0]*R_j[i,0]
            G[2*i,1]=2*((R_i[i,0]*R_i[i,1])-(R_j[i,0]*R_j[i,1]))
            G[2*i+1,1]=R_i[i,1]*R_j[i,0]+R_i[i,0]*R_j[i,1]
            G[2*i,2]=2*((R_i[i,0]*R_i[i,2])-(R_j[i,0]*R_j[i,2]))
            G[2*i+1,2]=R_i[i,2]*R_j[i,0]+R_i[i,0]*R_j[i,2]

            G[2*i,3]=(R_i[i,1]**2)-(R_j[i,1]**2)
            G[2*i+1,3]=R_i[i,1]*R_j[i,1]
            G[2*i,4]=2*((R_i[i,2]*R_i[i,1])-(R_j[i,2]*R_j[i,1]))
            G[2*i+1,4]=R_i[i,2]*R_j[i,1]+R_i[i,1]*R_j[i,2]
            G[2*i,5]=(R_i[i,2]**2)-(R_j[i,2]**2)
            G[2*i+1,5]=R_i[i,2]*R_j[i,2]
        GU,GS,GVh=np.linalg.svd(G)
        Q_sq=np.zeros((3,3))
        GV=GVh[-1,:]
        Q_sq[0, 0] = GV[0]
        Q_sq[0, 1] = GV[1]
        Q_sq[1, 0] = GV[1]
        Q_sq[0, 2] = GV[2]
        Q_sq[2, 0] = GV[2]
        Q_sq[1, 1] = GV[3]
        Q_sq[2, 1] = GV[4]
        Q_sq[1, 2] = GV[4]
        Q_sq[2, 2] = GV[5]
        Q_sq=np.abs(Q_sq)
        Q=np.linalg.cholesky(Q_sq)
        R_true = np.dot(R_hat, Q)
        S_true = np.dot(np.linalg.inv(Q), S_hat)
        X = S_true[0, :]
        Y = S_true[1, :]
        Z = S_true[2, :]
        # print(S_true.shape)
        factor = 1
        pointcloud = np.zeros((X.shape[0], 3))
        pointcloud[:, 0] = X
        pointcloud[:, 1] = Y
        pointcloud[:, 2] = Z * factor
        if self.show_scatter_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            pnt3d = ax.scatter(pointcloud[:, 0],  # x
                               pointcloud[:, 1],  # y
                               pointcloud[:, 2],  # z
                               c='b',
                               marker="o")
            plt.title(f'Structure from Motion')
            plt.show()
        with open('term_project.ply', 'w') as f:
            # Write the PLY header
            f.write('ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'.format(pointcloud.shape[0]))
            # Write the point coordinates from the matrix
            for i in range(pointcloud.shape[0]):
                f.write('{} {} {}\n'.format(pointcloud[i, 0], pointcloud[i, 1], pointcloud[i, 2]))

if __name__ == "__main__":
    path_to_images = r'C:\Users\udayr\PycharmProjects\CVfiles\term_project\hotel\hotel'
    imgs_arr = []
    seq_files = [f for f in os.listdir(path_to_images) if f.startswith('hotel.seq')]
    seq_numbers=[int(f.split('.')[1].split('seq')[-1]) for f in seq_files]
    seq_files=[f for _, f in sorted(zip(seq_numbers, seq_files))]
    for img_name in seq_files:
        img = cv2.imread(path_to_images + '\\' + img_name, 0)
        imgs_arr.append(np.asarray(img).astype(float))

    corner_images=SFM(imgs_arr,show_tracked_points=False,show_scatter_plot=True).point_coords()
