import numpy as np # linear algebra
import cv2
import os


def calculate_scores(a, b):
    gp = np.sum(a)  # y_true中标记为篡改的像素数
    tp = np.sum(a * b)  # y_true和y_pred中标记不一致的像素数
    pp = np.sum(b)  # y_pred中标记为篡改的像素数

    precision = tp / pp
    recall = tp / gp

    f1 = (2 * precision * recall) / (precision + recall)

    intersection = np.sum(a * b)  # 交集像素数
    union = np.sum(a) + np.sum(b) - intersection  # 并集像素数
    iou = intersection / union

    #final_score = f1 + iou
    final_score = f1

    return final_score

def traverse_image_files(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(root, file)
                image_files.append(file_path)
    return image_files

# image = cv2.imread("dataset_Casia/new/IMG__mask/Sp_S_CND_A_pla0016_pla0016_0196.jpg")
# # 将图像转换为NumPy数组
# image_a_array = np.array(image)
# image_a_array = np.divide(image_a_array, 255.0)
#
# image = cv2.imread("dataset_Casia/TRUE_MASK/Sp_S_CND_A_pla0016_pla0016_0196_gt.png")
# image_b_array = np.array(image)
# image_b_array = np.divide(image_b_array, 255.0)
#
# score = calculate_scores(image_a_array,image_b_array)
#
# print(score)
total_score = 0
for file_name in os.listdir("dataset_Casia/IMG_mask"):
    file_name = file_name.split()[0]
    path_1 = os.path.join("dataset_Casia/IMG_mask",file_name+".jpg")
    path_2 = os.path.join("dataset_Casia/TRUE_MASK",file_name+"_gt.png")

    image = cv2.imread(path_1)
    # 将图像转换为NumPy数组
    image_a_array = np.array(image)
    image_a_array = np.divide(image_a_array, 255.0)

    image = cv2.imread(path_2)
    image_b_array = np.array(image)
    image_b_array = np.divide(image_b_array, 255.0)

    score = calculate_scores(image_a_array,image_b_array)
    total_score= total_score + score
print(total_score/10.0)