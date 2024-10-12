CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --dataset coco  --model psp --jpu --aux  --epoch 200 --backbone resnet50 \
	               --lr 0.001 --checkname  Updated_Crop_480_lr_defualt_PSP_res50_FPAB-Afford_train
  

