
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

if [ "$#" -ge 3 ]; then
    tsy="$3"
else
    tsy="False"
fi


if [ "$#" -ge 4 ]; then
    original="$4"
else
    original="True"
fi

# if [ "$#" -ge 3 ]; then
#     permute_docs="$3"
# else
#     permute_docs='False'
# fi

# model_path="./run_saves/tsy_join_method_${mode}_${par_id}/"
# echo "$model_path"

# resume_ckpt="./pegasus/summ_checkpoints/step=28110-vloss=2.05-avgr=0.3091.ckpt"

# --resume_ckpt ${resume_ckpt} \
CUDA_VISIBLE_DEVICES=${gpu} nohup python test_hf_primer.py --par_id ${par_id} \
    --tsy ${tsy} --original ${original} > test_hf_primer_${par_id}_${tsy}_${original}.out 2>&1 &



# CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main-modify.py --mode ${mode} > primer-${mode}_${par_id}.out 2>&1 &

