import os
import sys
from indoor3d_util import DATA_PATH, collect_point_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   #E:\PythonProgram\LCHNet\semantic_seg\utils
ROOT_DIR = os.path.dirname(BASE_DIR) #E:\PythonProgram\LCHNet\semantic_seg\  项目根目录
sys.path.append(BASE_DIR)

#anno_paths的值为各个数据集地址的列表
anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
        collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        print(anno_path, 'ERROR!!')
