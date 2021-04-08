# Copyright (c) Open-MMLab. All rights reserved.
import pdb 
import cv2 
import numpy as np 
import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt 

from mmcv.image import imread, imwrite 
from .color import color_val 

colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255), 
             (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), 
             (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), 
             (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), 
             (255, 0, 0), (0, 0, 255)] 
             
edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
         [3, 5], [4, 6], [5, 6], [5, 7], 
         [7, 9], [6, 8], [8, 10], [5, 11], 
         [6, 12], [11, 12], [11, 13], 
         [13, 15], [12, 14], [14, 16]] 
         
ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), 
      (255, 0, 0), (0, 0, 255), (255, 0, 255), 
      (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255), 
      (255, 0, 0), (0, 0, 255), (255, 0, 255),
      (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)] 

def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def imshow_bboxes(img,
                  bboxes,
                  colors='green',
                  top_k=-1,
                  thickness=1,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        colors (list[str or tuple or Color]): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.
    """
    img = imread(img)

    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        _bboxes = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, colors[i], thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)

def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)

def imshow_extremes(img, 
                    bboxes, 
                    extremes, 
                    labels, 
                    class_names=None, 
                    score_thr=0, 
                    out_file=None): 
    assert bboxes.ndim == 2 
    assert labels.ndim == 1 
    assert bboxes.shape[0] == labels.shape[0] 
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5 
    img = imread(img) 
    
    im      = img[:, :, (2, 1, 0)] 
    fig, ax = plt.subplots(figsize=(12, 12)) 
    fig     = ax.imshow(im, aspect='equal') 
    plt.axis('off') 
    fig.axes.get_xaxis().set_visible(False) 
    fig.axes.get_yaxis().set_visible(False) 

    if score_thr > 0: 
        assert bboxes.shape[1] == 5 
        scores = bboxes[:, -1] 
        inds = scores > score_thr 
        bboxes = bboxes[inds, :] 
        extremes = extremes[inds, :] 
        labels = labels[inds]
   
    for bbox, label, extreme in zip(bboxes, labels, extremes): 
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1]) 
        right_bottom = (bbox_int[2], bbox_int[3]) 
        poly = np.array([[extreme[0],extreme[1]],[extreme[2],extreme[3]], \
                         [extreme[4],extreme[5]],[extreme[6],extreme[7]]], np.int32) 
        poly = poly.reshape((-1,1,2))
        
        ax.add_patch(plt.Rectangle((bbox_int[0], bbox_int[1]), 
                                    bbox_int[2]-bbox_int[0], bbox_int[3]-bbox_int[1], 
                                    fill=False, edgecolor= 'g', linewidth=2.0))

        poly = np.array([[extreme[0], extreme[1]], [extreme[2], extreme[3]], \
                         [extreme[4], extreme[5]], [extreme[6], extreme[7]]], np.int32)

        color = np.random.rand(3)
        ax.add_patch(plt.Polygon(poly, fill=True, color=color, alpha=0.5, edgecolor=None))
        ax.add_patch(plt.Polygon(poly, fill=False, edgecolor='w', linewidth=1.0))

        label_text = class_names[
              label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'     

        ax.text(bbox_int[0]+1, bbox_int[1]-3, label_text, bbox=dict(facecolor='g', ec='g',
                                                                    lw=0, alpha=0.5),
                                                                    fontsize=10, color='white', weight='bold')    

    if  out_file is not None:
        plt.savefig(out_file)
        plt.savefig(out_file.replace('jpg', 'pdf'))
        plt.cla()
        plt.close('all')     

def imshow_polygons(img, 
                    bboxes, 
                    polygons, 
                    labels, 
                    class_names=None, 
                    score_thr=0, 
                    out_file=None): 
    assert bboxes.ndim == 2 
    assert labels.ndim == 1 
    assert bboxes.shape[0] == labels.shape[0] 
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5 
    img = imread(img) 
    
    im      = img[:, :, (2, 1, 0)] 
    fig, ax = plt.subplots(figsize=(12, 12)) 
    fig     = ax.imshow(im, aspect='equal') 
    plt.axis('off') 
    fig.axes.get_xaxis().set_visible(False) 
    fig.axes.get_yaxis().set_visible(False) 

    if score_thr > 0: 
        assert bboxes.shape[1] == 5 
        scores = bboxes[:, -1] 
        inds = scores > score_thr 
        bboxes = bboxes[inds, :] 
        polygons = polygons[inds, :] 
        labels = labels[inds]
   
    for bbox, label, polygon in zip(bboxes, labels, polygons): 
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1]) 
        right_bottom = (bbox_int[2], bbox_int[3]) 
        poly = polygon.reshape(-1, 2).astype(np.int32)
        
        color = np.random.rand(3)
        ax.add_patch(plt.Polygon(poly, fill=True, color=color, alpha=0.5, edgecolor=None))
        ax.add_patch(plt.Polygon(poly, fill=False, edgecolor='w', linewidth=1.0))

        label_text = class_names[
              label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'       

    if out_file is not None:
       plt.savefig(out_file)
       plt.savefig(out_file.replace('jpg', 'pdf'))
       plt.cla()
       plt.close('all')     


def imshow_pose(img, 
                bboxes, 
                kps, 
                labels, 
                class_names=None, 
                score_thr=0, 
                out_file=None): 
    assert bboxes.ndim == 2 
    assert labels.ndim == 1 
    assert bboxes.shape[0] == labels.shape[0] 
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5 
    img = imread(img) 
    
    im      = img[:, :, (2, 1, 0)] 
    fig, ax = plt.subplots(figsize=(12, 12)) 
    fig     = ax.imshow(im, aspect='equal') 
    plt.axis('off') 
    fig.axes.get_xaxis().set_visible(False) 
    fig.axes.get_yaxis().set_visible(False) 

    if score_thr > 0: 
        assert bboxes.shape[1] == 5 
        scores = bboxes[:, -1] 
        inds = scores > score_thr 
        bboxes = bboxes[inds, :] 
        kps = kps[inds, :] 
        labels = labels[inds]
   
    for bbox, label, kp in zip(bboxes, labels, kps): 
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1]) 
        right_bottom = (bbox_int[2], bbox_int[3]) 
        kp = kp.reshape(-1, 2)

        ax.add_patch(plt.Rectangle((bbox_int[0], bbox_int[1]), 
                                    bbox_int[2]-bbox_int[0], bbox_int[3]-bbox_int[1], 
                                    fill=False, edgecolor= 'mediumslateblue', linewidth=2.0))
        
        for i in range(kp.shape[0]):
            kp_x = kp[i, 0]
            kp_y = kp[i, 1]

            if colors_hp[i] == (255,0,255):
                color = 'magenta'
            elif colors_hp[i] == (255,0,0):
                color = 'blue'
            else: #(0, 0, 255)
                color = 'red'
            
            plt.scatter(kp_x, kp_y, color = color, s = 40)

            for j, e in enumerate(edges):
                if kp[e].min() > 0:
                    if ec[j] == (255, 0, 255):
                        color = 'magneta'
                    elif ec[j] == (255, 0, 0):
                        color = 'blue'
                    else: #(0, 0, 255)
                        color = 'red'
                    
                    plt.plot([kp[e[0], 0], kp[e[1], 0]],
                             [kp[e[0], 1], kp[e[1], 1]], color=color, linewidth=4)

            label_text = 'person'
            label_text += f'|{bbox[-1]:.02f}'
            ax.text(bbox_int[0], bbox_int[1]-2, label_text, bbox=dict(facecolor='mediumslateblue', 
                                                                      ec='mediumslateblue',
                                                                      lw=0, alpha=0.5),
                                                                      fontsize=10, color='white', 
                                                                      weight='bold')

    if out_file is not None:
       plt.savefig(out_file)
       plt.savefig(out_file.replace('jpg', 'pdf'))
       plt.cla()
       plt.close('all')     
