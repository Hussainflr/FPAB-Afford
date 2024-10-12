CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset coco --model fcn --jpu --aux  --backbone resnet50\
	--crop-size 244 --base-size 300  --lr 0.001 --epoch 80 --checkname S_244lr001_dilatedFCN_res50_FPAB-Afford_train

