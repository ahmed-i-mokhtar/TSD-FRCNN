import cv2
model_name=''
import glob
import numpy as np

#video_name='/content/drive/MyDrive/Conti_dataset_images_front_resized'
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

out = cv2.VideoWriter('demo_set_0.mp4', fourcc, 3.0, (1500,640))
#nuscenes_sequential_frames_np = np.load('/home/khaled/3D_Object_Detection/Scene_Lyft_2/kitti_format/Lyft_sequential_frames_scene_2.npy')
#nuscenes_sequential_frames_list=nuscenes_sequential_frames_np.tolist()
#print(nuscenes_sequential_frames_list)
import os

frames_path='/home/amokhtar/teams/continental/mot/TCIS/swedish-outputs-2/'

frames_list=sorted(os.listdir(frames_path))
'''
for frame_name in nuscenes_sequential_frames_list:
    img=cv2.imread(f'{frames_path}{frame_name}_online.png')
    print(frame_name)  
    print(f'{frames_path}{frame_name}_online.png')  
    resized_image = cv2.resize(img, (1224, 1024)) 
    print(frame_name)
    out.write(resized_image)

out.release()
'''
'''
for frame_name in frames_list:
    #frame_name=f'Track_A-Sphere-{index+1}-2.jpg'
    img=cv2.imread(frames_path+frame_name)
    #resized_image = cv2.resize(img, (1080, 720)) 
    print(frame_name)
    out.write(img)

out.release()
'''
count =1

for frame_name in frames_list:
    img=cv2.imread(f'{frames_path}{frame_name}')
    # img = cv2.resize(img, (3537,1769)) 
    print(count)  
    count= count+1
    out.write(img)

out.release()
