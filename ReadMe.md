# SiamRPN-TF

* This repository includes a tensorflow implementation of  SiamRPN[1]
* The best AUC score on OTB100 of this repository so far: **0.593** （the AUC score in the paper is 0.637)



## Differences between this repository and SiamRPN[1]
* less training videos(unable to download YouTube-BB ...)
    * DET2014+VID2015+LASOT+GOT10k(about 20k videos) 
    * SiamRPN[1] use YouTube-BB+VID2015（about 100k videos)
* more data augmentation
    * random mixup [2]
    * random down-sample image resolution
    * random image blur

Add more details ... (TODO)



### reference

[1] High Performance Visual Tracking With Siamese Region Proposal Network (CVPR 2018)

[2] Bag of Freebies for Training Object Detection Neural Networks

