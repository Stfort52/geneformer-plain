MASTER_ADDR=$(hostname -i)
MASTER_PORT=$1
NODES=$2 # comma separated list of nodes
TO_RUN=$3

if [ -z "$MASTER_PORT" ] || [ -z "$NODES" ] || [ -z "$TO_RUN" ]; then
    echo "Usage: $0 <master_port> <nodes> <to_run>"
    exit 1
fi

WORLD_SIZE=$(echo $NODES | tr ',' '\n' | wc -l)

rank=1
export NODE_RANK=0
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE

for node in $(echo $NODES | sed "s/,/ /g"); do
    if [ $node == $(hostname) ]; then
        echo "Starting master on $MASTER_ADDR:$MASTER_PORT with $WORLD_SIZE nodes"
        python3 -m masters.train.$TO_RUN &
        continue
    fi
    
    echo "Starting slave on $node with rank $rank"
    ssh $node \
        NODE_RANK=$rank \
        MASTER_ADDR=$MASTER_ADDR \
        MASTER_PORT=$MASTER_PORT \
        WORLD_SIZE=$WORLD_SIZE \
        "$(which python3) -m masters.train.$TO_RUN" &
    
    rank=$((rank+1))

done

wait

