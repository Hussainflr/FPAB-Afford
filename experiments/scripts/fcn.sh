CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --dataset coco --model fcn --jpu --aux  --backbone resnet50 \
	--crop-size 480 --base-size 520  --lr 0.001 --epoch 200 --checkname Updated_Crop_lr001_dilatedFCN_res50_FPAB-Afford_train

