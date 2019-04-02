# SiamRPN-TF

* This repository includes a tensorflow implementation of  SiamRPN[1]
* The best AUC score on OTB100 of this repository so far: **0.602** （the AUC score in the paper[1] is 0.637)

## Some differences between this repository and SiamRPN
* less training videos(unable to download YouTube-BB :dizzy_face:...)
    * DET2014+VID2015+LASOT+GOT10k(about 20k videos) 
    * SiamRPN[1] use YouTube-BB+VID2015（about 100k videos)
* more data augmentation
    * random mixup [2]
    * random image blur & color jittering

## Train & Test

Add more details ... (TODO)

## Things I find important in the experiment

* Training dataset is more important than anything else
* Use pretrained feature extraction and train the network from deep layer to shallow layer step by step
  * first train deeper layers with larger learning rate
  * and then train the whole network with smaller learning rate
* Models trained from scratch always perform worse (best AUC score on OTB100: 0.564) in my experiment even with longer training time(5 times), this might be attributed to limited training dataset

## Progress(ongoing)
* [ ] Achieve the AUC score in the paper[1]
* [ ] To speed up training by replace tf.py_func & numpy  operation with pure tf implementation 
* [x] Multi-GPU Training & load dataset with lmdb [lmdb dataset download(about 120G)  code: dn58](https://pan.baidu.com/s/1vhx-G_ctqsxtglwFAu9qNQ)
* [x] Convert [pytorch pretrained model](https://pan.baidu.com/s/1OTseQUknI6EgXddcPt8s6A#list/path=%2F) to initialize embedding function
* [x] leaning rate warmup
* [x] Train network with static images and dynamic videos
* [x] Add time decay option to weight loss[3] (a slight improvement)
* [x] Add random Mixup[2]

### Reference

[1] High Performance Visual Tracking With Siamese Region Proposal Network (CVPR 2018)

[2] Bag of Freebies for Training Object Detection Neural Networks

[3] Learning Attentions: Residual Attentional Siamese Network for High Performance Online Visual Tracking

[4] [SiamFC tensorflow implementation](https://github.com/bilylee/SiamFC-TensorFlow)

