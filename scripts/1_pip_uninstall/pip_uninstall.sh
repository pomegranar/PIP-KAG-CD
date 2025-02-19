cd ../src/1_uninstall

python3 1_2_pip_uninstall.py \
    --pruning_start_layer # Starting Layer for Pruning  \
    --pruning_nums # Number of layers to prune.  \
    --pretrained_model_path # Path to the pre-trained model to be pruned. \
    --save_path # Save path for the model after pruning.


    