CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --dataset coco --model fcn_vgg --aux  --backbone vgg16 \
	--crop-size 480 --base-size 520  --lr 0.001 --epoch 300 --checkname Updated_Crop_lr_001_FCN32_VGG_FPAB-Afford_train
