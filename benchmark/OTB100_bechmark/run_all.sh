#/bin/bash
total_step=1562500
step=10000
bash run.sh SiamRPN_1e6 5000 ${step} ${total_step}
bash run.sh SiamRPN_1e6_fixedconv3 5000 ${step} ${total_step}

bash run.sh SiamRPN_1e6_warmup 5000 ${step} ${total_step}
bash run.sh SiamRPN_1e6LRe-4 5000 ${step} ${total_step}
bash run.sh SiamRPN_1e6LRe-4_dist1000 5000 ${step} ${total_step} 

bash run.sh SiamRPN_TRI_1e6 5000 ${step} ${total_step} 

#bash run.sh SiamRPN_TRI_norm_ftall 5000 ${step} ${total_step} 
#bash run.sh SiamRPN_ftall 5000 ${step} ${total_step} 

#bash run.sh SiamRPN_TRI_norm 5000 5000 625000 
#bash run.sh SiamRPN_weightedMixup 5000 5000 625000 
#bash run.sh SiamRPN_weightedLabelMixup 5000 5000 625000 
#bash run.sh SiamRPN_woMixUp 5000 5000 625000 
#bash run.sh SiamRPN 5000 5000 625000 
#bash run.sh SiamRPN_stage2 5000 5000 625000 
#bash run.sh SiamRPNG 5000 5000 625000 
#bash run.sh SiamRPN_IOULoss 5000 5000 625000 
#bash run.sh SiamRPN_IOULoss_stage2 5000 5000 625000
#bash run.sh SiamRPN_fixedEmbedding 5000 5000 625000
#bash run.sh SiamRPN-mixup 5000 5000 625000
#bash run.sh SiamRPN_tri 5000 5000 625000

#bash run.sh SiamRPN_GIOU 5000 5000 625000
#bash run.sh SiamRPN_Scratch 5000 5000 625000
#bash run.sh SiamRPNG2 5000 5000 625000
#bash run.sh SiamRPN_TRI_combined 5000 5000 625000
#bash run.sh SiamRPN 5000 5000 625000
#bash run.sh SiamRPN-lr1 5000 5000 625000
