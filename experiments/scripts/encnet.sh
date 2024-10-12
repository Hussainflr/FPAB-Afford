CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --dataset coco \
	       --model encnet --jpu --aux --se-loss  --lr 0.001 --epochs 80\
     --backbone resnet50 --checkname Updated_Crop_Jpu_encnet_res50_FPAB-Afford_train

