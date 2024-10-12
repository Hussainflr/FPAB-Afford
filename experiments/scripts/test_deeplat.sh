CUDA_VISIBLE_DEVICES=4 python test.py --dataset coco --model deeplab   --aux     --backbone resnet50  --split val \
 --resume /BigDisk/hussain/code/segmentation/FastFCN/experiments/segmentation/runs/coco/deeplab/Updated_Crop_No-jpu_deeplab_imgS480_res50_FPAB-Afford_train/model_best.pth.tar \
 --mode test --test-batch-size 1 --save-folder Deeplab_Updated_Crop_No-jpu-deeplab_res50
