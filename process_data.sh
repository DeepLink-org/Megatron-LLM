python tools/preprocess_data.py --input=/mnt/lustrenew/shanhang/project/Megatron-LLM/output.json \
    --output_prefix=./pdopt/70B80layers \
    --tokenizer_type=SentencePieceTokenizer \
    --vocab_file=/mnt/lustrenew/share_data/PAT/datasets/llama2/Llama-2-70b/tokenizer.model \
    --chunk_size=1 \
    --workers=1 \
    --no_new_tokens