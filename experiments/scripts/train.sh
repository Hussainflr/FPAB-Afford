 CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset coco --lr 0.001 --model encnet --jpu --aux --se-loss --backbone resnet50 --checkname encnetRes50_FPAB-Afford_train
