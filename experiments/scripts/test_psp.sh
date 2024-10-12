CUDA_VISIBLE_DEVICES=5 python test.py --dataset coco --model psp  --jpu --aux     --backbone resnet50  --split val \
	  --resume /BigDisk/hussain/code/segmentation/FastFCN/experiments/segmentation/runs/coco/psp/Updated_Crop_480_lr_defualt_PSP_res50_FPAB-Afford_train/model_best.pth.tar \
	    --mode test --test-batch-size 1 --save-folder Updated_Crop_psp_res50
