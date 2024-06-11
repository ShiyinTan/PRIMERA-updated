if [ "$#" -ge 1 ]; then
    mode="$1"
else
    mode="test"
fi

# 设置多个devices：0,1
if [ "$#" -ge 2 ]; then
    gpu="$2"
else
    gpu=0
fi

if [ "$#" -ge 3 ]; then
    par_id="$3"
else
    par_id="0"
fi

if [ "$#" -ge 4 ]; then
    permute_docs="$4"
else
    permute_docs='False'
fi

output_file="primer-${mode}_${par_id}.out"

host_name=$(hostname)
current_time=$(date)
model_path="./run_saves/tsy_join_method_${mode}_${par_id}/"
echo "$model_path on $host_name at $current_time. " > $output_file



if [ $mode = "test" ]; then
    resume_ckpt="./run_saves/tsy_join_method_train_5/summ_checkpoints/step=30921-vloss=2.03-avgr=0.3146.ckpt"
    # --resume_ckpt ${resume_ckpt} \
    CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main_modify.py --mode ${mode} \
                --model_path ${model_path} --beam_size 5 --batch_size 1 --strategy ddp \
                --permute_docs ${permute_docs} >> $output_file 2>&1 &
else
    CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main_modify.py --mode ${mode} \
                --model_path ${model_path} --beam_size 5 --batch_size 1 --strategy ddp \
                --permute_docs ${permute_docs} >> $output_file 2>&1 &
fi



# CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main-modify.py --mode ${mode} > primer-${mode}_${par_id}.out 2>&1 &

