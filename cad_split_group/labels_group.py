#!/usr/bin/python
#encoding:utf-8

#!/usr/bin/python
#encoding:utf-8
from PIL import Image             ##调用库
import os
import os.path
import numpy as np
#该py文件用于匹配ROI的中心，确定move和pose位置,输出对应排序txt文件


def Preprocess(N):
  f1 = '../cad_split_with_pose/cad_flow_train_split_%d.txt'%(N) # txt文件和当前脚本在同一目录下，所以不用写具体路径
  f2 = './cad_flow_train_split_%d.txt'%(N)
	
	#person2group labelshift
  switch_dict={'1':2,
							'2':1,
							'3':2,
							'4':3,
							'5':1,
							'6':4,
							'7':5,
							'8':1}

  with open(f1, 'r+') as file_to_read:
    while True:
      lines1 = file_to_read.readline() # 整行读取数据
      if not lines1:
        break
        pass
      lines1=lines1.strip()
      lines_split=lines1.split()
      move=lines_split[-2]
      group_move=switch_dict.get(move)
      lines1=lines1+' %s'%(str(group_move))
      #print(lines1)
      with open(f2, 'a+') as file_to_write:
        file_to_write.write(lines1+'\n') # 整行读取数据

      
  
  print("split%d successfully preprocessed!"%(N))

# im = Image.open("./1.jpg")  ##文件存在的路径
# box=(300,100,700,700)
# region=im.crop(box)
# region.save("./2.jpg")
# region.show()

def Concatenate(N):
  with open('../labels/track%d.txt'%(N), 'r') as fa:
    with open(r'../labels_move_pose/track%d.txt'%(N), 'r') as fb:
      with open('../labels_preprocessed/track%d.txt' % (N), 'w') as fc:
        for line in fa:
          fc.write(line.strip('\r\n')+' ')
          fc.write(fb.readline())


if __name__ =='__main__':
  for N in range(1,4):
    Preprocess(N)
    print("All Clear!")
