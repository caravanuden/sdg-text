nohup python3 -u doc2vec_encode_data.py &
process_id=$!
wait $process_id
nohup python3 -u doc2vec_encode_data.py --epochs=10 &
