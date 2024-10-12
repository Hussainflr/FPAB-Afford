CUDA_VISIBLE_DEVICES=4 python test.py --dataset coco --model fcn  --jpu --aux     --backbone resnet50  --split val \
           --resume /BigDisk/hussain/code/segmentation/FastFCN/experiments/segmentation/runs/coco/fcn/Updated_Crop_lr001_dilatedFCN_res50_FPAB-Afford_train/model_best.pth.tar \
             --mode test --test-batch-size 1 --save-folder FCN-Jpu-Updatedlr001-Crop_psp_res50

