#!/bin/bash

module load openmpi/gcc/64/4.1.5
module load cuda12.2/toolkit/12.2.2
#module load anaconda3/3.11
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/home/matthewn/projects/GPT_Diagnosis/llama3
MAIN_MACHINE="sabercore-a100-001"
MAIN_PROCESS_IP=$(host $MAIN_MACHINE | cut -d" " -f 4)
HOST="$MAIN_MACHINE,sabercore-a100-002,sabercore-a100-003"
N_MACHINE=$(echo $HOST | tr ',' '\n' | wc -l)
N_PROCESS=$N_MACHINE

echo $N_PROCESS
echo $N_MACHINE
echo $MAIN_PROCESS_IP
echo $MAIN_MACHINE

mpirun -np $N_PROCESS \
  --output-filename mpi_out \
  --prefix /cm/shared/apps/openmpi4/gcc/4.1.5 \
  --host $HOST \
  -x MAIN_PROCESS_IP=$MAIN_PROCESS_IP \
  -x N_MACHINE=$N_MACHINE \
  -x N_PROCESS=$N_PROCESS \
  -x CUDA_HOME \
  -wd $WORKDIR \
  bash -c '
    /home/matthewn/.conda/envs/kuda/bin/accelerate launch \
    --config_file ./accelerate_config_multigpu_brady.yaml \
    --multi_gpu \
    --main_process_ip $MAIN_PROCESS_IP \
    --num_machines $N_MACHINE \
    --num_processes $N_PROCESS \
    --machine_rank $OMPI_COMM_WORLD_RANK \
    --mixed_precision bf16 \
    run_llama3.py
  '

    #run_llama3_multi_gpu_accel.py --model_name "meta-llama/Meta-Llama-3-70B-Instruct" --instruction "Who was the Comanche Quanah Parker?" --output_dir ./output

#bash -c '/home/matthewn/.conda/envs/kuda/bin/accelerate launch \
#--config_file ./accelerate_config_multigpu_brady.yaml \
#--multi_gpu \
#--main_process_ip $MAIN_PROCESS_IP \
#--num_machines $N_MACHINE \
#--num_processes $N_PROCESS \
#--machine_rank $OMPI_COMM_WORLD_RANK \
#--mixed_precision fp16 \
#run_llama3_multi_gpu_accel.py'
