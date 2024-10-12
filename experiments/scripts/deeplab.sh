CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --dataset coco --lr 0.001 --model deeplab   --aux  --batch-size 16 --epoch 80 \
       	--backbone resnet50 \
	 --crop-size 480 --base-size 580  \
	--checkname Updated_Crop_deeplab_imgS480_res50_FPAB-Afford_train

