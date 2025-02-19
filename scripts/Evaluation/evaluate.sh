cd /src/3_evaluate/

to_ex_path=evaluation_confiqa_vllm_acc.py
eval_files=/data/CoConflictQA

to_eval_model=/home/liuzhenghao/hpc/project/lc_rag/experiments/sampled_3w2_from_train/ex/23_6_e3_margin1/checkpoint-2100/merged
output_dir=/home/liuzhenghao/hpc/project/lc_rag/experiments/sampled_3w2_from_train/ex/eval_res/23_6_e3_margin1/test_
device=0
model_type=llama3_pruning_ffn   # llama3_pruning_ffn llama3


max_new_tokens=32
use_chat_template=true


cur_model=${to_eval_model}
for file in ${eval_files}/*.jsonl
do

    filename=$(basename "$file")
    CUDA_VISIBLE_DEVICES=$device python3 $to_ex_path \
        --model_name $cur_model \
        --data_path $file \
        --schema base \
        --output_path ${output_dir}/${filename}_res.json \
        --log_path ${output_dir}${filename}.log \
        --model_type $model_type \
        --use_chat_template $use_chat_template \
        --max_new_tokens $max_new_tokens
done
