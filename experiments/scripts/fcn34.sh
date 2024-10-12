CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset coco --model fcn   --backbone resnet34 --head \
	--crop-size 256 --base-size 320  --lr 0.001 --epoch 150 --batch-size 8 --checkname Crop_Lr001_DFCN_r34_FPAB-Afford_train

