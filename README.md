# PCBind

## Build Environment 

use requirments.txt

## Prepare Data

run construction_PDBbind_training_and_test_dataset.ipynb
run generate_5_conformer.py

## Train models and evaluation

### For DDP
```
python -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 2 --master_addr=localhost --master_port=12345 main.py -d 0 -m 1 --batch_size 4 --label ddp --use_equivalent_native_y_mask --lr 0.0001 --recycling_num 8 --distributed True --gpu 2 --max_node 1000

```

### For single server

```
python main.py -d 0 -m 1 --batch_size 4 --label single --use_equivalent_native_y_mask --lr 0.0001 --recycling_num 8

```
## Checkpoint of our trained model

download from: https://drive.google.com/file/d/1o2lo4pQH1yG_M9P0KyfjX8PeAneHmkGN/view?usp=drive_link




