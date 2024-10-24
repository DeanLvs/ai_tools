PRE_SEQ_LEN=128
LR=5e-5
NUM_GPUS=1

nohup torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file /nvme0n1-disk/bigLLM/merged_train_t.json \
    --validation_file /nvme0n1-disk/bigLLM/merged_val_t.json \
    --preprocessing_num_workers 64 \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path /nvme0n1-disk/bigLLM/chatglm2-6b \
    --output_dir /nvme0n1-disk/bigLLM/chatglm2/checkpoint \
    --overwrite_output_dir \
    --max_source_length 384 \
    --max_target_length 384 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --fp16 \
    &

