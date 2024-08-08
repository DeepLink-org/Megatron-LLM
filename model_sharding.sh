python tools/checkpoint_util.py \
    --target_tensor_parallel_size 8 \
    --target_pipeline_parallel_size 4 \
    --load_dir /mnt/lustrenew/shanhang/project/Megatron-LLM/megatron_weights/ \
    --save_dir /mnt/lustrenew/shanhang/project/Megatron-LLM/shard_weights/ \
    --model_type llama2 \
    --true_vocab_size 32000 \
    --megatron_path /mnt/cache/shanhang/newhome/project/fork/Megatron-LLM