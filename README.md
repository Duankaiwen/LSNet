## Attention！I failed to open source the trained model! You may need to train by yourselves.
# [Location-Sensitive Visual Recognition with Cross-IOU Loss](xxx)

by [Kaiwen Duan](https://scholar.google.com/citations?hl=zh-CN&user=TFHRaZUAAAAJ&scilu=&scisig=AMD79ooAAAAAXLv9_7ddy26i4c6z5n9agk05m97faUdN&gmla=AJsN-F78W-h98Pb2H78j6lTKbjdn0fklhe2X_8CCPqRU2fC4KJEIbllhD2c5F0irMR3zDiehKt_SH26N2MHI1HlUMw6qRba9HMbiP3vnQfJqD82FrMAPdlU&sciund=10706678259143520926&gmla=AJsN-F5cOpNUdnI6YrZ9joRa6JE2nP6wFKU1GKVkNIfCmmgjk431Lg2BYCS6wn5WWZxdnzBjLfaUwdUJtvPXo53vfoOQoTGP5fHh2X0cCssVtXm8BI4PaM3_oQvKYtCx7o1wivIt1l49sDK6AZPvHLMxxPbC4GbZ1Q&sciund=10445692451499027349), [Lingxi Xie](http://lingxixie.com/Home.html), [Honggang Qi](http://people.ucas.ac.cn/~hgqi), [Song Bai](http://songbai.site/), [Qingming Huang](https://scholar.google.com/citations?user=J1vMnRgAAAAJ&hl=zh-CN) and [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN)

**The code to train and evaluate the proposed LSNet is available here. For more technical details, please refer to our [arXiv paper](xxx).**

<div align=center>
<img src=https://github.com/Duankaiwen/LSNet/blob/main/code/resources/LSVR.png width = "500" height = "350" alt="" align=center />
</div>

## Abstract

  Object detection, instance segmentation, and pose estimation are popular visual recognition tasks which require localizing the object by internal or boundary landmarks. This paper summarizes these tasks as location-sensitive visual recognition and proposes a unified solution named location-sensitive network (LSNet). Based on a deep neural network as the backbone, LSNet predicts an anchor point and a set of landmarks which together define the shape of the target object. The key to optimizing the LSNet lies in the ability of fitting various scales, for which we design a novel loss function named cross-IOU loss that computes the cross-IOU of each anchor-landmark pair to approximate the global IOU between the prediction and groundtruth. The flexibly located and accurately predicted landmarks also enable LSNet to incorporate richer contextual information for visual recognition. Evaluated on the MSCOCO dataset, LSNet set the new state-of-the-art accuracy for anchor-free object detection (a 53.5% box AP) and instance segmentation (a 40.2% mask AP), and shows promising performance in detecting multi-scale human poses. 

**If you encounter any problems in using our code, please contact Kaiwen Duan: kaiwenduan@outlook.com**

## Performances
<div align=center>
<img src=https://github.com/Duankaiwen/LSNet/blob/main/code/resources/bbox.png width = "200" height = "400" alt="" align=center />
</div>
##

##
<div align=center>
<img src=https://github.com/Duankaiwen/LSNet/blob/main/code/resources/segm.png width = "500" height = "450" alt="" align=center />
</div>
##

##
![Graph](https://github.com/Duankaiwen/LSNet/blob/main/code/resources/pose.png)
##

##
![Graph](https://github.com/Duankaiwen/LSNet/blob/main/code/resources/visualization.png)
##

##
![Graph](https://github.com/Duankaiwen/LSNet/blob/main/code/resources/compare.png)
##
