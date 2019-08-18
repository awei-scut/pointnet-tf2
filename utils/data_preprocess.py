import tensorflow as tf
from tensorflow import keras
from plyfile import PlyData, PlyElement, make2d, PlyParseError, PlyProperty
import numpy as np
import os
import sys
import h5py
# h5py 用来操作hdf5文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


## 对点云进行采样 path
SAMPLING_BIN = os.path.join(BASE_DIR, 'third_party/mesh_shampling/build/pcsample')

SAMPLING_POINT_NUM = 2048
SAMPLING_LEAF_SIZE = 0.005

MODELNET40_PATH = '../datasets/modelnet40'

# 导出ply
def export_ply(pc, filename):
    vertex = np.zeros(pc.shape[0], dtype=[('x','f4'), ('y','f4'),('z','f4')])
    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][0], pc[i][2])
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
    ply_out.write(filename)

# 对obj采样 返回命令行语句
def get_sampling(obj_filename, ply_filename):
    cmd = SAMPLING_BIN + ' ' + obj_filename
    cmd += ' ' + ply_filename
    cmd += ' -n_samples %d ' % SAMPLING_POINT_NUM
    cmd += ' -leaf_size %f ' % SAMPLING_LEAF_SIZE
    return cmd


# -----------------------------
# load MODELNET4 shapes

def get_catagory_names():
    shape_names_file = os.path.join(MODELNET40_PATH, 'shape_names.txt')
    shape_names = [line.rstrip() for line in open(shape_names_file)]
    return shape_names

def get_obj_filenames():
    obj_filelist_file = os.path.join(MODELNET40_PATH, 'filelist.txt')
    obj_filenames = [os.path.join(MODELNET40_PATH, line.rstrip()) for line in open(obj_filelist_file)]
    print('Got %d obj files in modelnet40.' % len(obj_filenames))
    return obj_filenames


# Helper function to create the father folder and all subdir folders if not exist
def batch_mkdir(output_folder, subdir_list):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for subdir in subdir_list:
        if not os.path.exists(os.path.join(output_folder, subdir)):
            os.mkdir(os.path.join(output_folder, subdir))

# 保存datasets
def save_h5_data_label_normal(h5_filename, data, label, normal,
                              data_dtype='float32',
                              label_dtype='unit8',
                              normal_dtype='float32'):
    h5_out = h5py.Flie(h5_filename)
    h5_out.create_dataset(
        'data', data=data,
        compression='gzip', compression_opts=4,
        dtype=data_dtype)
    h5_out.create_dataset(
        'normal', data=normal,
        compression='gzip', compression_opts=4,
        dtype=normal_dtype)
    h5_out.create_dataset(
        'label', data=label,
        compression='gzip', compression_opts=1,
        dtype=label_dtype)
    h5_out.close()


# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

# Read numpy array data and label from h5_filename
def load_h5_data_label_normal(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return (data, label, normal)


# Read numpy array data and label from h5_filename
def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

# Load PLY file
def load_ply_data(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Load PLY file
def load_ply_normal(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['normal'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array


