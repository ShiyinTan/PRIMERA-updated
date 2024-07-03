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
# if [ "$filter_score" -ne '0.0' ]; then
#     model_path="./run_saves/tsy_join_method_${mode}_${par_id}_${filter_score}/"
# else
#     model_path="./run_saves/tsy_join_method_${mode}_${par_id}/"
# fi
model_path="./run_saves/tsy_join_method_${mode}_${par_id}/"
echo "$model_path on $host_name at $current_time. " > $output_file



# steps to add:
## 1. add it on function __getitem__ of SummarizationDataset at dataloader.py 
## 2. add join_method to primer_summarizer_module.py
# join_method:
## concat_start_wdoc_global 
## tsy_design
## no_rand_sentence
## global_rand_sentence
## indoc_rand_sentence
## sim_sent_transformer
## indoc_sim_sent_transformer
## only_drop_lowsim_sent
if [ $mode = "test" ]; then
    # resume_ckpt="./run_saves/tsy_join_method_train_5/summ_checkpoints/step=5622-vloss=1.96-avgr=0.3212.ckpt"
    # resume_ckpt="./run_saves/tsy_join_method_train_only_drop_lowsim_sent_0.3/summ_checkpoints/step=5622-vloss=2.02-avgr=0.3190.ckpt"
    # resume_ckpt="./run_saves/tsy_join_method_train_indoc_sim_sent_transformer_0.3/summ_checkpoints/step=14055-vloss=2.16-avgr=0.3155.ckpt"
    # resume_ckpt="./run_saves/tsy_join_method_train_sim_sent_transformer_0.3/summ_checkpoints/step=8433-vloss=2.06-avgr=0.3121.ckpt"
    # resume_ckpt="./run_saves/tsy_join_method_train_sim_sent_transformer/summ_checkpoints/step=19677-vloss=2.22-avgr=0.3143.ckpt"
    # resume_ckpt="./run_saves/tsy_join_method_train_indoc_sim_sent_transformer/summ_checkpoints/step=5622-vloss=1.98-avgr=0.3139.ckpt"
    # resume_ckpt="./run_saves/tsy_join_method_train_indoc_rand_sentence/summ_checkpoints/step=19677-vloss=2.19-avgr=0.3115.ckpt"
    resume_ckpt="./run_saves/tsy_join_method_train_global_rand_sentence/summ_checkpoints/step=11244-vloss=2.04-avgr=0.3064.ckpt"
    # --resume_ckpt ${resume_ckpt} \
    CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main_modify.py --mode ${mode} \
                --model_path ${model_path} --beam_size 5 --batch_size 1 --strategy auto \
                --permute_docs ${permute_docs} --join_method global_rand_sentence \
                --filter_score 0.0 --resume_ckpt ${resume_ckpt} >> $output_file 2>&1 &
else
    CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main_modify.py --mode ${mode} \
                --model_path ${model_path} --beam_size 5 --batch_size 16 --strategy auto \
                --permute_docs ${permute_docs} --join_method only_drop_lowsim_sent \
                --filter_score 0.3 >> $output_file 2>&1 &
fi



# CUDA_VISIBLE_DEVICES=${gpu} nohup python primer_hf_main-modify.py --mode ${mode} > primer-${mode}_${par_id}.out 2>&1 &

