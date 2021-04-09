## Attention！I failed to open source the trained model! You may need to train by yourselves.
# [Location-Sensitive Visual Recognition with Cross-IOU Loss](xxx)

by [Kaiwen Duan](https://scholar.google.com/citations?hl=zh-CN&user=TFHRaZUAAAAJ&scilu=&scisig=AMD79ooAAAAAXLv9_7ddy26i4c6z5n9agk05m97faUdN&gmla=AJsN-F78W-h98Pb2H78j6lTKbjdn0fklhe2X_8CCPqRU2fC4KJEIbllhD2c5F0irMR3zDiehKt_SH26N2MHI1HlUMw6qRba9HMbiP3vnQfJqD82FrMAPdlU&sciund=10706678259143520926&gmla=AJsN-F5cOpNUdnI6YrZ9joRa6JE2nP6wFKU1GKVkNIfCmmgjk431Lg2BYCS6wn5WWZxdnzBjLfaUwdUJtvPXo53vfoOQoTGP5fHh2X0cCssVtXm8BI4PaM3_oQvKYtCx7o1wivIt1l49sDK6AZPvHLMxxPbC4GbZ1Q&sciund=10445692451499027349), [Lingxi Xie](http://lingxixie.com/Home.html), [Honggang Qi](http://people.ucas.ac.cn/~hgqi), [Song Bai](http://songbai.site/), [Qingming Huang](https://scholar.google.com/citations?user=J1vMnRgAAAAJ&hl=zh-CN) and [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN)

**The code to train and evaluate the proposed LSNet is available here. For more technical details, please refer to our [arXiv paper](xxx).**

<div align=center>
<img src=https://github.com/Duankaiwen/LSNet/blob/main/code/resources/lsvr.png width = "600" height = "250" alt="" align=center />
  
*The location-sensitive visual recognition tasks, including object detection, instance segmentation, and human pose estimation, can be formulated into localizing an anchor point (in red) and a set of landmarks (in green). Our work aims to offer a unified framework for these tasks.*
</div>

## Abstract

  Object detection, instance segmentation, and pose estimation are popular visual recognition tasks which require localizing the object by internal or boundary landmarks. This paper summarizes these tasks as location-sensitive visual recognition and proposes a unified solution named location-sensitive network (LSNet). Based on a deep neural network as the backbone, LSNet predicts an anchor point and a set of landmarks which together define the shape of the target object. The key to optimizing the LSNet lies in the ability of fitting various scales, for which we design a novel loss function named cross-IOU loss that computes the cross-IOU of each anchor-landmark pair to approximate the global IOU between the prediction and groundtruth. The flexibly located and accurately predicted landmarks also enable LSNet to incorporate richer contextual information for visual recognition. Evaluated on the MSCOCO dataset, LSNet set the new state-of-the-art accuracy for anchor-free object detection (a 53.5% box AP) and instance segmentation (a 40.2% mask AP), and shows promising performance in detecting multi-scale human poses. 

**If you encounter any problems in using our code, please contact Kaiwen Duan: kaiwenduan@outlook.com**

## Bbox AP(%) on COCO test-dev
|Method          |  Backbone       | epoch | MS<sub>train<sub> |  AP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| :------------- | :-------:       | :---: | :---------------: | :--: | :-------------: | :-------------: | :------------: | :------------: | :------------: | 
|                |   
| *Anchor-based:*|   
|Libra R-CNN     | X-101-64x4d     | 12  |      N            | 43.0 |     64.0        |       47.0      |      25.3      |      45.6      |      54.6      |
| AB+FSAF*       | X-101-64x4d     | 18  |      Y            | 44.6 |     65.2        |       48.6      |      29.7      |      47.1      |      54.6      |
| FreeAnchor*    | X-101-32x8d     | 24  |      Y            | 47.3 |     66.3        |       51.5      |      30.6      |      50.4      |      59.0      |
| GFLV1*         | X-101-32x8d     | 24  |      Y            | 48.2 |     67.4        |       52.6      |      29.2      |      51.7      |      60.2      |
| ATSS*          | X-101-64x4d-DCN | 24  |      Y            | 50.7 |     68.9        |       56.3      |      33.2      |      52.9      |      62.4      |
| PAA*           | X-101-64x4d-DCN | 24  |      Y            | 51.4 |     69.7        |       57.0      |      34.0      |      53.8      |      64.0      |
| GFLV2*         | R2-101-DCN      | 24  |      Y            | 53.3 |     70.9        |       59.2      |      35.7      |      56.1      |      65.6      |
| YOLOv4-P7*     | CSP-P7          | 450 |      Y            | 56.0 |     73.3        |       61.2      |      38.9      |      60.0      |      68.6      |
|                |   
| *Anchor-free:* |
| ExtremeNet     | HG-104          |200  |      Y            | 43.2 |     59.8        |       46.4      |      24.1       |     46.0      |      57.1      | 
|RepPointsV1      R-101-DCN 24 Y 46.5 67.4 50.9 30.3 49.7 57.1
             
<div align=center>
<img src=https://github.com/Duankaiwen/LSNet/blob/main/code/resources/segm.png width = "450" height = "400" alt="" align=center />
</div>

<hr/>
<hr/>

![Graph](https://github.com/Duankaiwen/LSNet/blob/main/code/resources/pose.png)

<hr/>
<hr/>

![Graph](https://github.com/Duankaiwen/LSNet/blob/main/code/resources/visualization.png)

<hr/>
<hr/>

<div align=center>
<img src=https://github.com/Duankaiwen/LSNet/blob/main/code/resources/compare.png width = "500" height = "700" alt="" align=center />
 </div>
 
<hr/>
<hr/>
