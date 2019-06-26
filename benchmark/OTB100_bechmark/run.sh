#/bin/bash

config_name=$1
start_iter=$2
step=$3
end_iter=$4
for i in $(seq ${start_iter} ${step} ${end_iter})
do
  echo "testing model: $config_name iter_num: $i"
  model_path="/home/lab-xiong.jiangfeng/Projects/SiameseRPN/Logs/${config_name}/track_model_checkpoints/${config_name}/model.ckpt-${i}.meta"
  result_dir="/home/lab-xiong.jiangfeng/Projects/SiameseRPN/tracker_benchmark/results/OPE/${config_name}-${i}/scores_tb100"
  echo ${model_path}
  echo ${result_dir}

  if [ ! -f ${model_path} -o -d ${result_dir} ];then
  echo "model file not exist or resulti_dir exits, skip"
  continue
  fi

  echo tb100 | python run_trackers.py -t SiamRPN -s tb100 -e OPE -m ${config_name} -c $i & 
  let count+=1
  sleep 1m
  gpu=`python ../scripts/wait_gpu.py`
  
  until [ "$gpu" -ne "-1" ]; do
    gpu=`python ../scripts/wait_gpu.py`
  done
done
wait
