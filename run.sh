#!/bin/bash
NUM_CLIENTS=3
TRAIN_EPOCHS=30
NUM_ROUNDS=5

echo "Starting server"
python server.py --num_rounds=${NUM_ROUNDS}&
sleep 3  # Sleep for 3s to give the server enough time to start

for ((i=0;i<${NUM_CLIENTS};i++)); do
    echo "Starting client $i"
    python client.py \
        --partition=${i} \
        --train_epochs=${TRAIN_EPOCHS} \
        --num_partition=${NUM_CLIENTS} &

    # python client.py \
    #     --partition=${i} \
    #     --train_epochs=${TRAIN_EPOCHS} \
    #     --num_partition=${NUM_CLIENTS} \
    #     --log_tensorboard &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait