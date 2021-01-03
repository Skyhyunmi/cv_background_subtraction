import numpy as np
import cv2 as cv
import os
def make(input_path, gt_path, result_path):
    #input_path='/ori_in'
    #gt_path='ori_bg'
    #result_path='ori_res'
     ##### Set path
    #input_path = './input' # input path
    #gt_path = './backgroundtruth' # groundtruth path
    #result_path = './result'  # result path

    groundtruth = [img for img in sorted(os.listdir(gt_path)) if img.endswith(".png")]

        ##### your result image
    result_image = [img for img in sorted(os.listdir(result_path)) if img.endswith(".png")]

    #assert len(groundtruth) == len(result_image), "Result should have the same number of samples as groundtruth"

        ##### variables for confusion matrix
    TP = 0   # True Positive
    TN = 0   # True Negative
    FP = 0   # False Positive
    FN = 0   # False Negative
    GT = 0   # Number of GroundTruth Pixels
    EST = 0  # Number of Estimated Pixels

        ##### make confusion matrix for images
    for index in range(len(groundtruth)):
        #if index%1000000000000000==10000000000000:
         # print('TP: %.3f'%TP)
          #print('FP: %.3f'%FP)
          #print('GT: %.3f'%GT)
          #print(index)
            ##### groundtruth pre-processing
        gt = cv.imread(os.path.join(gt_path, groundtruth[index]))
        gt_gray = cv.cvtColor(gt, cv.COLOR_BGR2GRAY)
        gt_cvt = np.where(gt_gray >82, 255, 0)
        gt_cvt_rev = np.where(gt_gray<=82, 255, 0)
        # cv.imwrite(os.path.join("./test", 'gray%06d.png' % (index + 1)), gt_cvt)

            ##### result pre-processing
        result = cv.imread(os.path.join(result_path, result_image[index]))
        result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

        result_gray = np.where(result_gray ==255, 255, 0)
        # cv.imwrite(os.path.join("./test", 'gray%06d.png' % (index + 1)), result_gray)
            ##### variable summation

        TP += np.sum(np.logical_and(gt_cvt, result_gray).astype(np.uint8))
        FP += np.sum(np.logical_and(gt_cvt_rev, result_gray).astype(np.uint8))
        GT += np.sum(gt_cvt)/255
        if index <800:
            TP=0
            FP=0
            GT=0
    print('TP : %d  ||  GT : %d ||  EST : %d' % (TP, GT, TP + FP))
    print('Recall : %.3f %%  ||  Precision : %.3f %%' %
      (((100.0 * TP) / GT), ((100.0 * TP) / (TP + FP))))
    print(' ')