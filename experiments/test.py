

import os
import sys
sys.path.append('../../')
import numpy as np
import torch
import torchvision.transforms as transform
import cv2
import encoding.utils as utils

from PIL import Image
from tqdm import tqdm

from torch.utils import data
from matplotlib import pyplot as plt
from encoding.nn import BatchNorm2d
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule

from option import Options



def apply_mask( image, mask, color=None, alpha=0.5):
        """Apply the given mask to the image.
        """
        image = np.asarray(image)
        mask = np.asarray(mask)
        if color is not None:
            for c in range(3):
                print(list(color[c]))
                image[:, :, c] = np.where(mask == 1,
                                          image[:, :, c] *
                                          (1 - alpha) + alpha * ( 255, 208, 25),
                                          image[:, :, c])
            return image
        else:
            image =  image *(1.0 - alpha) + mask * alpha
        #image = np.uint8(image * 255)
        #print("Type: ", type(image))
        #image = Image.fromarray(image)
        return image
def denormalize(tensor, mean, std):

    for t, m, s in zip(tensor, mean, std):

        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return tensor





def test(args):
    # output folder
    outdir = 'results/'+args.save_folder
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    testset = get_segmentation_dataset(args.dataset, split=args.split, mode=args.mode,
                                       transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated,
                                       lateral = args.lateral, jpu = args.jpu, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = BatchNorm2d,
                                       base_size = args.base_size, crop_size = args.crop_size)
        # resuming checkpoint
        #if args.resume is None or not os.path.isfile(args.resume):
        #    raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    print(model)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    if not args.ms:
        scales = [1.0]
    evaluator = MultiEvalModule(model, testset.num_class, scales=scales, flip=args.ms).cuda()
    
    evaluator.eval()
    metric = utils.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)

    for i, (image, dst) in enumerate(tbar):
         
        if 'val' in args.mode or 'train' in args.mode:
            with torch.no_grad():
                predicts = evaluator.parallel_forward(image)
                metric.update(dst, predicts)
                pixAcc, mIoU, IoU = metric.get()
             
                tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f, background: %.3f, grasp: %.3f, wrap-g: %3.f, contain: %.3f,\
                        openable: %.3f, cleanable: %.3f, drinkable: %.4f, \
                        readable: %.3f, scoop: %3.f,' % (pixAcc, mIoU, IoU[0],IoU[1],IoU[2],IoU[3] , IoU[4],IoU[5],IoU[6],IoU[7], IoU[8]))
        else:
            with torch.no_grad():
                outputs = evaluator.parallel_forward(image)
                predicts = [testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                            for output in outputs]
                
            for predict, impath in zip(predicts, dst):
                
                mask = utils.get_mask_pallete(predict, args.dataset)
                #print(type(mask), type(image[0])) 
                mask = mask.convert('RGB')
               
                #image = image[0].cpu().numpy() 
                #print(image.shape)
                #img = (image-image.min())/(image.max()-image.min())

                
                
                #img = np.transpose(img, (1,2,0))
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                #cv2.imwrite(os.path.join(outdir,str(i)+'.jpeg'), img)
                #exit()
                
                #plt.figure()
                image =  denormalize(image[0], [.485, .456, .406], [.229, .224, .225])
                image = transform.ToPILImage()(image)
                #print(image.size, mask.size)
                #plt.imshow(image, interpolation='nearest')
                #plt.savefig("svae.png")
                #exit()
                #print(type(image))
                #mask = np.asarray(mask)
                #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
                #print(image.shape, mask.shape)
                maskedimage = Image.blend(image, mask, 0.5)
                maskedimage.save(os.path.join(outdir, 'bl_'+str(i)+'.jpeg'))
                #cv2.addWeighted(mask, 0.5, image, (0.5), 0.0)
                #cv2.imwrite(os.path.join(outdir,'blended'+str(i)+'.png'), maskedimage)
                #outname = os.path.splitext(impath)[0] + '.png'
                #maskname = str(i)+'.jpeg'
                #mask.save(os.path.join(outdir, maskname))
                #image.save(os.path.join(outdir,'ori_'+str(i)+'.jpg'))

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    test(args)
