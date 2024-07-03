
# 设置多个devices：0,1
if [ "$#" -ge 1 ]; then
    gpu="$1"
else
    gpu=0
fi

if [ "$#" -ge 2 ]; then
    par_id="$2"
else
    par_id=0
fi


# if [ "$#" -ge 3 ]; then
#     permute_docs="$3"
# else
#     permute_docs='False'
# fi

# model_path="./run_saves/tsy_join_method_${mode}_${par_id}/"
# echo "$model_path"

# resume_ckpt="./pegasus/summ_checkpoints/step=28110-vloss=2.05-avgr=0.3091.ckpt"

outfile=test_hf_primer_custom_dataset_${par_id}.out
host_name=$(hostname)
echo "Running model on $host_name:($gpu)" > $outfile

# --resume_ckpt ${resume_ckpt} \
CUDA_VISIBLE_DEVICES=${gpu} nohup python -u test_hf_primer_custom_dataset.py --par_id ${par_id} \
        >> $outfile 2>&1 &


