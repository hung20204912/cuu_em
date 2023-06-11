python ./program/ease_deepsmells/main.py \
    --model "DeepSmells" \
    --nb_epochs  60 \
    --train_batchsize 128 \
    --valid_batchsize 128 \
    --lr 0.027 \
    --threshold 0.5 \
    --hidden_size_lstm 100 \
    --data_path "..\embedding-dataset\cuBERT\LongMethod_CuBERT_embeddings.pkl" \
    --tracking_dir ".\tracking\cuBERT" \
    --result_dir ".\result" \

# Here is the configure of the model
# Note: Set the configuration - Try to edit the directory like above example
# data_path: the path of the embedding file
# tracking_dir: 
# result_dir: the path of the result file
