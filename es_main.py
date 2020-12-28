# -*- coding: utf-8 -*-

import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

input_path = './input'
gt_path = './groundtruth'
result_path = './result_es'


def test(points, thres, dist):
    print('=' * 50)
    print("points:", points)
    print("threshold:", thres)
    print("distance:", dist)
    print('=' * 50)

    ##### params
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.01))
    dx_alpha = 0.7
    dy_alpha = 0.7
    horizontal_crop = 30
    vertical_crop = 20

    ##### Load input
    input_list = [img for img in sorted(os.listdir(input_path)) if img.endswith('jpg')]

    ##### Initialize prev/current frame
    frame_current = cv.imread(os.path.join(input_path, input_list[0]))
    frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.uint8)
    frame_prev_gray = frame_current_gray

    ##### recording
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    out = cv.VideoWriter('./foreground.avi', fourcc, 30.0, (720, 480))

    ##### for iteration
    cnt = 0

    ##### frame difference queue
    dx_queue = []
    dy_queue = []

    ##### Append dx, dy
    x_li = []
    y_li = []
    sx_li = []
    sy_li = []

    ##### background subtractor
    fgbg = cv.createBackgroundSubtractorMOG2()
    cv.BackgroundSubtractorMOG2.clear(fgbg)
    cv.BackgroundSubtractorMOG2.setShadowValue(fgbg, 255)

    for idx in range(len(input_list)):
        cnt += 1

        ##### feature detection
        corners_prev = cv.goodFeaturesToTrack(frame_prev_gray, points, thres, dist)

        ##### optical flow of previous frame to current frame
        corners_current, st, err = cv.calcOpticalFlowPyrLK(frame_prev_gray, frame_current_gray, corners_prev, None,
                                                           **lk_params)
        good_current = corners_current[st == 1]
        good_prev = corners_prev[st == 1]

        ##### estimate transform matrix
        transform = cv.estimateAffine2D(good_prev, good_current, False)[0]

        ##### in case of transform == None
        if transform is None:
            transform = prev_transform

        prev_transform = transform

        ##### transform matrix between two frames
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])

        x_li.append(dx)
        y_li.append(dy)

        ##### for smoothing dx, dy
        dx_queue.append(dx)
        dy_queue.append(dy)

        if len(dx_queue) > 2:
            dx_queue.pop(0)
        if len(dy_queue) > 2:
            dy_queue.pop(0)

        ##### transform, inverse-transform matrix for stabilizing
        new_transform = np.zeros((2, 3), np.float32)

        new_transform[0, 0] = np.cos(-da)
        new_transform[0, 1] = -np.sin(-da)
        new_transform[1, 0] = np.sin(-da)
        new_transform[1, 1] = np.cos(-da)

        inv_new_transform = np.zeros((2, 3), np.float32)

        inv_new_transform[0, 0] = np.cos(da)
        inv_new_transform[0, 1] = -np.sin(da)
        inv_new_transform[1, 0] = np.sin(da)
        inv_new_transform[1, 1] = np.cos(da)

        ##### smoothed dx, dy
        if len(dx_queue) > 1:
            smoothed_dx = (dx_queue[0] * (1 - dx_alpha) + dx_queue[1] * dx_alpha)
            new_transform[0, 2] = - smoothed_dx
            inv_new_transform[0, 2] = smoothed_dx
            sx_li.append(smoothed_dx)

        else:
            new_transform[0, 2] = - dx
            inv_new_transform[0, 2] = dx
            sx_li.append(dx)

        if len(dy_queue) > 1:
            smoothed_dy = (dy_queue[0] * (1 - dy_alpha) + dy_queue[1] * dy_alpha)
            new_transform[1, 2] = - smoothed_dy
            inv_new_transform[1, 2] = smoothed_dy
            sy_li.append(smoothed_dy)

        else:
            new_transform[1, 2] = - dy
            inv_new_transform[1, 2] = dy
            sy_li.append(dy)

        ##### warp current frame based on extracted transform matrix
        warped_current_gray = cv.warpAffine(frame_current_gray, new_transform, (720, 480))

        ##### estimate foreground region
        fgmask = fgbg.apply(warped_current_gray, learningRate=0.0007)
        fgmask = cv.medianBlur(fgmask.astype(np.uint8), 11)

        ##### rewarp inversely
        fgmask = cv.warpAffine(fgmask, inv_new_transform, (720, 480))
        fgmask_mk2 = np.where(fgmask > 0, 255.0, 0.0)

        result = fgmask_mk2.astype(np.uint8)
        result[:vertical_crop, :] = 0
        result[-vertical_crop:, :] = 0
        result[:, -horizontal_crop:] = 0
        result[:, :horizontal_crop] = 0

        ##### make result file
        cv.imwrite(os.path.join(result_path, 'result%06d.png' % (idx + 1)), result)
        out.write(cv.cvtColor(result, cv.COLOR_GRAY2BGR))

        ##### renew background
        #     background = background * (1-alpha) + warped_current_gray * alpha
        if idx + 1 == len(input_list):
            break
        ##### renew prev, cur
        frame_prev_gray = frame_current_gray
        frame_current = cv.imread(os.path.join(input_path, input_list[idx + 1]))
        frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.uint8)

        ##### show iteration
        if cnt % 100 == 0:
            print('iter:', cnt)
            plt.imshow(result)

    out.release()
    cv.destroyAllWindows()

    ##### gt, result list
    gt_list = [img for img in sorted(os.listdir(gt_path)) if img.endswith(".png")]
    result_list = [img for img in sorted(os.listdir(result_path)) if img.endswith(".png")]

    ##### variables for confusion matrix
    TP = 0  # True Positive
    TN = 0  # True Negative
    FP = 0  # False Positive
    FN = 0  # False Negative
    GT = 0  # Number of GroundTruth Pixels
    EST = 0  # Number of Estimated Pixels

    ##### make confusion matrix for images
    for index in range(len(gt_list)):
        if index < 800:
            continue

        if index % 500 == 499: print('iter:', index)

        ##### groundtruth pre-processing
        gt = cv.imread(os.path.join(gt_path, gt_list[index]))
        gt_gray = cv.cvtColor(gt, cv.COLOR_BGR2GRAY)
        gt_cvt = np.where(gt_gray > 82, 255, 0)
        gt_cvt_rev = np.where(gt_gray <= 82, 255, 0)

        gt_cvt[:vertical_crop, :] = 0
        gt_cvt[-vertical_crop:, :] = 0
        gt_cvt[:, -horizontal_crop:] = 0
        gt_cvt[:, :horizontal_crop] = 0

        gt_cvt_rev[:vertical_crop, :] = 0
        gt_cvt_rev[-vertical_crop:, :] = 0
        gt_cvt_rev[:, -horizontal_crop:] = 0
        gt_cvt_rev[:, :horizontal_crop] = 0

        ##### result pre-processing
        result = cv.imread(os.path.join(result_path, result_list[index]))
        result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        result_gray = np.where(result_gray == 255, 255, 0)

        ##### variable summation
        TP += np.sum(np.logical_and(gt_cvt, result_gray).astype(np.uint8))
        FP += np.sum(np.logical_and(gt_cvt_rev, result_gray).astype(np.uint8))
        GT += np.sum(gt_cvt) / 255

    print('TP : %d  ||  GT : %d ||  EST : %d' % (TP, GT, TP + FP))
    print('Recall : %.3f %%  ||  Precision : %.3f %%' %
          (((100.0 * TP) / GT), ((100.0 * TP) / (TP + FP))))


test(500, 0.005, 10)