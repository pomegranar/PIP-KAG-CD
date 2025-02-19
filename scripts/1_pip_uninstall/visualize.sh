cd ../src/1_uninstall

CUDA_VISIVLE_DEVICES=0 python3 1_1_visualize_neuron_inhibition_ratio.py \
    --in_file_path ../data/func_data/draw_acctivations_shuf1k.jsonl \
    --visualize_path $visualize_res.png$  \
    --pretrained_model_path $path of the pretrained model$ \
    --model_type $type of the pretrained model$


