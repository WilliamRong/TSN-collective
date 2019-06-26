#!/usr/bin/python
#encoding:utf-8

import os  
      
def gci(filepath):
#遍历filepath下所有文件，包括子目录
  for fpathe,dirs,fs in os.walk(filepath):
    for f in dirs:
      print os.path.join(fpathe,f)

def Labels_ready(track_list):
  labels_list=[]
  for track_name in track_list:
    track_split=track_name.split('_')
    seq_idx=int(track_split[0][5:])
    person_idx=int(track_split[1])
    move_label=int(track_split[2][:-4])
    count=open('./labels_ready/seq%.2d/%s'%(seq_idx,track_name)).readlines()
    count_split=count[0].split()
    start_frame=int(count_split[1])
    end_frame=len(count)+start_frame-1

    pose_label=int(count_split[-1])
    pose_count=0
    for i in range(0,len(count)):
      pose_count+=1
      if(pose_label!=int(count[i].split()[-1])):
        end_frame= start_frame+pose_count-2
        
        labels_list.append('./extracted_rgb_flow_cad_preprocessed/seq%.2d/crop_seq%.2d_%d %d %d %d %d'%(seq_idx,seq_idx,person_idx,start_frame,end_frame,move_label,pose_label))
        pose_label=int(count[i].split()[-1])
        start_frame=int(count[i].split()[1])
        pose_count=0
      if(i==len(count)-1):
        end_frame= start_frame+pose_count
        labels_list.append('./extracted_rgb_flow_cad_preprocessed/seq%.2d/crop_seq%.2d_%d %d %d %d %d'%(seq_idx,seq_idx,person_idx,start_frame,end_frame,move_label,pose_label))

  f_write=open('./labels_ready.txt','w+')
  for lines in labels_list:
    f_write.write(lines+'\n')
  print('labels clear!')

def file_name(file_dir):   
  files_list=[]
  for root, dirs, files in os.walk(file_dir):  
    #print(root) #当前目录路径  
    #print(dirs) #当前路径下所有子目录  
    #print files #当前路径下所有非目录子文件
    files_list.extend(files)
  return files_list


if __name__ =='__main__':
  #gci('./')
  Labels_ready(file_name('./labels_ready/'))  
  print("All Clear!")
