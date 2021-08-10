import os
import glob
import pandas as pd
import cv2
import csv

from PIL import Image


def readandwrite_csv(data_txt=None, data_csv=None, name=None, wh=None):
    data_txt.write(str(name) + ' ')  # 檔名 + 空格
    with open(data_csv) as csvFile:
        rows = csv.reader(csvFile)
        for row in rows:  # csv檔所有的點
            x1 = int(row[0]) - wh  # 左上x
            y1 = int(row[1]) - wh  # 左上y
            x2 = int(row[0]) + wh  # 右下x
            y2 = int(row[1]) + wh  # 右下y
            # 0是類別
            data_txt.write(str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(0) + ' ')

        train_f.write('\n')  # 換行


def readandwrite_txt(data_txt=None, input_txt=None, name=None, height=None, weight=None):
    data_txt.write(str(name) + ' ')  # 檔名 + 空格
    row_number = 1
    f = open(input_txt)
    max_num = len(f.readlines())
    f.close()
    with open(input_txt) as txtFile:

        for row in txtFile.readlines():  # 逐行讀txt檔所有的點

            s = row.split(' ')  # 用空格分開
            classes = int(s[0])  # 所屬類別
            center_x = float(s[1]) * weight
            center_y = float(s[2]) * height
            boundary_w = float(s[3]) * weight
            boundary_h = float(s[4]) * height

            x1 = int(center_x - boundary_w/2)  # 左上x
            y1 = int(center_y - boundary_h/2)  # 左上y
            x2 = int(center_x + boundary_w/2)  # 右下x
            y2 = int(center_y + boundary_h/2)  # 右下y

            if row_number == max_num:
                data_txt.write(str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(classes))
            else:
                data_txt.write(str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(classes) + ' ')
            row_number += 1

        data_txt.write('\n')  # 換行


# GT image path & csv
#train_path = 'C:/Users/Lab722 BX/Desktop/term project/dataset v4/train/images/'     # training圖片
val_path = 'C:/Users/Lab722 BX/Desktop/term project/dataset v4/test/images/'         # validation圖片
save_img_path = 'C:/Users/Lab722 BX/Desktop/term project/dataset v4/test/new_label/'  # 儲存圖片路徑

#train_imgFileList = sorted(glob.glob(train_path + '*.png'))
# train_csvFileList = sorted(glob.glob(train_path + '*.csv'))
#train_txtFileList = sorted(glob.glob(train_path.replace('images', 'labels') + '*.txt'))

val_imgFileList = sorted(glob.glob(val_path + '*.png'))
# val_csvFileList = sorted(glob.glob(train_path + '*.csv'))
val_txtFileList = sorted(glob.glob(val_path.replace('images', 'labels') + '*.txt'))
#train_f = open('train.txt', 'w')
val_f = open('val.txt', 'w')
count = 0  # 檔名

if __name__ == "__main__":

    # train.txt-------------------------------------------------------------------------
    #print('training set')
    #for (train_img, train_txt) in zip(train_imgFileList, train_txtFileList):
    #    image = cv2.imread(train_img)
    #    count += 1
    #    readandwrite_txt(train_f, train_txt, count, image.shape[0], image.shape[1])
    #    cv2.imwrite(save_img_path + str(count) + '.jpg', image)  # 儲存圖片
    #    print(count)
    # val.txt-------------------------------------------------------------------------
    print('validation set')
    for (val_img, val_txt) in zip(val_imgFileList, val_txtFileList):
        image1 = cv2.imread(val_img)
        count += 1
        readandwrite_txt(val_f, val_txt, count, image1.shape[0], image1.shape[1])
        cv2.imwrite(save_img_path + str(count) + '.jpg', image1)  # 儲存圖片
        print(count)
    #train_f.close()
    val_f.close()
