import os
import numpy as np
import cv2
import requests
import sys

from PIL import Image
from io import BytesIO
from matplotlib import pyplot




def get_a_random_pair():
    idx = np.random.randint(0, L)
    return (os.path.join(manTraNet_dataDir, this) for this in sample_pairs[idx])
import sys
sys.path.append('src')
import modelCore


from datetime import datetime


def read_rgb_image(image_file):
    rgb = cv2.imread(image_file, 1)[..., ::-1]
    return rgb


def decode_an_image_array(rgb, manTraNet):
    x = np.expand_dims(rgb.astype('float32') / 255. * 2 - 1, axis=0)
    t0 = datetime.now()
    y = manTraNet.predict(x)[0, ..., 0]
    t1 = datetime.now()
    return y, t1 - t0


def decode_an_image_file(image_file, manTraNet):
    rgb = read_rgb_image(image_file)
    mask, ptime = decode_an_image_array(rgb, manTraNet)
    return rgb, mask, ptime.total_seconds()

# 随机测试八张来自他自身数据集的图片
def random_show():
    for k in range(8):
        # get a sample
        forged_file, original_file = get_a_random_pair()
        # load the original image just for reference
        ori = read_rgb_image(original_file)
        # manipulation detection using ManTraNet
        rgb, mask, ptime = decode_an_image_file(forged_file, manTraNet)
        # show results
        pyplot.figure(figsize=(15, 5))
        pyplot.subplot(131)
        pyplot.imshow(ori)
        pyplot.title('Original Image')
        pyplot.subplot(132)
        pyplot.imshow(rgb)
        pyplot.title('Forged Image (ManTra-Net Input)')
        pyplot.subplot(133)
        pyplot.imshow(mask, cmap='gray')
        pyplot.title('Predicted Mask (ManTra-Net Output)')
        pyplot.suptitle(
            'Decoded {} of size {} for {:.2f} seconds'.format(os.path.basename(forged_file), rgb.shape, ptime))
        pyplot.show()
def test_one_pair(original_path='original.png',forged_file='fake.png'):
    original_file=original_path
    forged_file = forged_file
    ori = read_rgb_image(original_file)
    # manipulation detection using ManTraNet
    rgb, mask, ptime = decode_an_image_file(forged_file, manTraNet)
    # show results
    pyplot.figure(figsize=(15, 5))
    pyplot.subplot(131)
    pyplot.imshow(ori)
    pyplot.title('Original Image')
    pyplot.subplot(132)
    pyplot.imshow(rgb)
    pyplot.title('Forged Image (ManTra-Net Input)')
    pyplot.subplot(133)
    pyplot.imshow(mask, cmap='gray')
    pyplot.title('Predicted Mask (ManTra-Net Output)')
    pyplot.suptitle(
        'Decoded {} of size {} for {:.2f} seconds'.format(os.path.basename(forged_file), rgb.shape, ptime))
    pyplot.show()




def dirs_test_and_save_mask(dirs_file_path):
    path_list = os.listdir(dirs_file_path)
    for path_1 in path_list:
        mask_dir_path = os.path.join(dirs_file_path,path_1+'_mask')
        os.makedirs(mask_dir_path)
        sub_path = os.path.join(dirs_file_path,path_1)
        img_name_list = os.listdir(sub_path)
        for img_name in img_name_list:
            img_path = os.path.join(sub_path,img_name)
            rgb, mask, ptime=decode_an_image_file(img_path,manTraNet)
            pyplot.imsave(os.path.join(mask_dir_path,img_name), mask, cmap='gray')


manTraNet_root = ''
manTraNet_srcDir = os.path.join(manTraNet_root, 'src')
sys.path.insert(0, manTraNet_srcDir)
manTraNet_modelDir = os.path.join(manTraNet_root, 'pretrained_weights')

manTraNet_dataDir = os.path.join(manTraNet_root, 'data')
sample_file = os.path.join(manTraNet_dataDir, 'samplePairs.csv')
assert os.path.isfile(sample_file), "ERROR: can NOT find sample data, check `manTraNet_root`"
with open(sample_file) as IN:
    sample_pairs = [line.strip().split(',') for line in IN.readlines()]
L = len(sample_pairs)
print("INFO: in total, load", L, "samples")

manTraNet = modelCore.load_pretrain_model_by_index(4, manTraNet_modelDir)
# ManTraNet Architecture
print(manTraNet.summary(line_length=120))
# Image Manipulation Classification Network
IMCFeatex = manTraNet.get_layer('Featex')
print(IMCFeatex.summary(line_length=120))
if __name__ == '__main__':

    #测试自定义的一组图片
    original_path='original.png'
    forged_file='fake.png'
    test_one_pair(original_path,forged_file)
    #test_one_pair()
    #随机展示八组图片的效果
    random_show()
    # fake = read_rgb_image('fake.png')
    # y1,y2=decode_an_image_array(fake,manTraNet)
    # print(y1)

    dirs_test_and_save_mask('datasets_s')