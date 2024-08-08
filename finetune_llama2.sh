export CUDA_DEVICE_MAX_CONNECTIONS=1
set -euxo pipefail
IFS="," read -ra gpus_arr <<< "$SLURM_STEP_GPUS"
gpus_per_node=${#gpus_arr[@]}
num_processes=$((SLURM_NNODES * gpus_per_node))
head_node_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


GPUS_PER_NODE="$gpus_per_node"
NNODES="$SLURM_NNODES"
NODE_RANK="$SLURM_PROCID"
MASTER_ADDR="$head_node_ip"
MASTER_PORT=29512
WORLD_SIZE=$((GPUS_PER_NODE*NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 50"
TRAIN_ARGS="--train_iters 1000 --lr_decay_style cosine --lr_warmup_iters 5 --lr 2e-5 --min_lr 2e-6"

torchrun $DISTRIBUTED_ARGS finetune.py \
    --num_layers 80 \
        --hidden_size 8192 \
        --num_attention_heads 64 \
        --num_attention_heads_kv 8 \
        --seq_length 2048 \
        --max_position_embeddings 4096 \
        --position_embedding_type rotary \
        --glu_activation swiglu \
        --ffn_hidden_size 28672\
        --use_rms_norm \
        --layernorm_epsilon 1e-6 \
        --no_tie_embed_logits \
        --make_vocab_size_divisible_by 128 \
    --tensor_model_parallel_size 8 \
    --pipeline_model_parallel_size 4 \
    --load /mnt/cache/shanhang/newhome/project/fork/Megatron-LLM/shard_weights \
    --tensorboard_dir /mnt/cache/shanhang/newhome/project/fork/Megatron-LLM/tensorboard \
    --data_path /mnt/cache/shanhang/newhome/project/fork/Megatron-LLM/pdopt/70B80layers_text_document \
    --model_name llama2 \
    --tokenizer_type SentencePieceTokenizer \
    --vocab_file=/mnt/lustrenew/share_data/PAT/datasets/llama2/Llama-2-70b/tokenizer.model \
    --attention_dropout 0.0 \
    --fp16 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --no_bias_gelu_fusion \
    --no_bias_dropout_fusion \
    --no_gradient_accumulation_fusion \
    --sequence_parallel \
    --recompute_granularity selective \
    --data_type gpt \
    --variable_seq_lengths \
    --use_checkpoint_args \
    --use_flash_attn \
    $LOG_ARGS $TRAIN_ARGS 