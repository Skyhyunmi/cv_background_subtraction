# -*- coding: utf-8 -*-

import os

import cv2 as cv
import numpy as np
import eval

input_path = './input'
gt_path = './groundtruth'
result_path = './result_es'

 ##### load input
input = [img for img in sorted(os.listdir(input_path)) if img.endswith(".jpg")]

 ##### first frame and first background
mog_list=[0.1,0.2,0.3,]
his_list=[0.02,0.04,0.06,0.08]
lr_list=[ 0.0004,0.0005,0.0006]
me_list=[5,7,9,11]
#######################
lk_params = dict( winSize  = (21,21),
                     maxLevel = 3,
                     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.01))
dx_alpha = 0.7
dy_alpha = 0.7
horizontal_crop = 3
vertical_crop = 2
cum_dx, cum_dy, cum_da = 0, 0, 0
dx_queue = []
dy_queue = []
x_li = []
y_li = []
sx_li = []
sy_li = []
#######################

def work(points, thres,dist):

    ##### background substraction

    frame_current = cv.imread(os.path.join(input_path, input[0]))
    frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.uint8)
    frame_prev_gray = frame_current_gray
    background = frame_prev_gray
    medi=11
    lr=0.0005
    fgbg = cv.createBackgroundSubtractorMOG2() ##원래 20
    cv.BackgroundSubtractorMOG2.setBackgroundRatio(fgbg,lr)
    cv.BackgroundSubtractorMOG2.clear(fgbg)
    cv.BackgroundSubtractorMOG2.setShadowValue(fgbg, 255)
    # for medi in me_list:
    for image_idx in range(len(input)):
        corners_prev = cv.goodFeaturesToTrack(frame_prev_gray, points, thres, dist)

        ##### optical flow of previous frame to current frame
        corners_current, st, err = cv.calcOpticalFlowPyrLK(frame_prev_gray, frame_current_gray, corners_prev, None, **lk_params)
        good_current = corners_current[st==1]
        good_prev = corners_prev[st==1]

        ##### estimate transform matrix
        # transform = cv.estimateRigidTransform(good_prev, good_current, False)
        transform = cv.estimateAffine2D(good_prev,good_current)
        transform=transform[0]
        ##### in case of transform == None
        prev_transform = transform
        if transform is None:
          transform = prev_transform



        ##### transform matrix between two frames
        dx = transform[0,2]
        dy = transform[1,2]
        da = np.arctan2(transform[1,0], transform[0,0])

        x_li.append(dx)
        y_li.append(dy)

        ##### for smoothing dx, dy
        dx_queue.append(dx)
        dy_queue.append(dy)

        if len(dx_queue)>2:
          dx_queue.pop(0)
        if len(dy_queue)>2:
          dy_queue.pop(0)

        ##### Trajectory

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
          smoothed_dx = (dx_queue[0] * (1-dx_alpha) + dx_queue[1] * dx_alpha)
          new_transform[0, 2] = - smoothed_dx
          inv_new_transform[0, 2] = smoothed_dx
          sx_li.append(smoothed_dx)

        else:
          new_transform[0, 2] = - dx
          inv_new_transform[0, 2] = dx
          sx_li.append(dx)

        if len(dy_queue) > 1:
          smoothed_dy = (dy_queue[0] * (1-dy_alpha) + dy_queue[1] * dy_alpha)
          new_transform[1, 2] = - smoothed_dy
          inv_new_transform[1, 2] = smoothed_dy
          sy_li.append(smoothed_dy)

        else:
          new_transform[1, 2] = - dy
          inv_new_transform[1, 2] = dy
          sy_li.append(dy)

        ##### warp current frame based on extracted transform matrix
        warped_current_gray = cv.warpAffine(frame_current_gray, new_transform, (720, 480))


        #frame_current = cv.imread(os.path.join(input_path, input[image_idx]))
        fgmask = fgbg.apply(warped_current_gray,learningRate=lr)#0.0007 좋음
        fgmask0 = fgbg.apply(warped_current_gray,learningRate=-1)
        fgmask1 = fgbg.apply(warped_current_gray,learningRate=lr+0.00005)
        fgmask2 = fgbg.apply(warped_current_gray,learningRate=lr-0.00001)
        fgmask3 = fgbg.apply(warped_current_gray,learningRate=lr-0.00005)
        fgmask=fgmask+fgmask0+fgmask1+fgmask3
        #fgmask = img_fill(fgmask)
        fgmask = cv.medianBlur(fgmask,7)
        #cv.imshow('a',fgmask)
        fgmask = cv.warpAffine(fgmask, inv_new_transform, (720, 480))
        #cv.imshow('b',fgmask)
        fgmask_mk2 = np.where(fgmask > 0, 255.0, 0.0)
        #cv.imshow('c',fgmask)


    #     result = current_gray_masked_mk2.astype(np.uint8)
        result = fgmask_mk2.astype(np.uint8)
        result[:vertical_crop, :] = 0
        result[-vertical_crop:, :] = 0
        result[:, -horizontal_crop:] = 0
        result[:, :horizontal_crop] = 0
        cv.imwrite(os.path.join(result_path, 'result%06d.png' % (image_idx + 1)), result)
      ##### end of input
        if image_idx == len(input) - 1:
            break
        frame_prev_gray = frame_current_gray
        frame_current = cv.imread(os.path.join(input_path, input[image_idx+1]))
        frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.uint8)

      ##### If you want to stop, press ESC key
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break


 ##### evaluation result
    eval.make(input_path, gt_path, result_path)

def img_fill(im_in):  # n = binary image threshold
    im_th = im_in

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    fill_image = im_th | im_floodfill_inv
    #print('%d %d'% (np.sum(fill_image)/256, h*w*0.4))

    if np.sum(fill_image)/255> h*w*0.17:
        return im_in
    return fill_image

if __name__ == '__main__':
    work(500, 0.005, 10)