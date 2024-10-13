# FPHA-Afford 
An object affordance dataset for human-object-robot interaction. <br> 
### Dataset
Please download dataset from google drive link.

 Gdrive: https://drive.google.com/drive/folders/1qj5e3j5K3EIRReCnmiji62vs1M0S6NX9?usp=sharing


### Trained Models Checkpoints 
Trained models checkpoints can be found in following link.<br>
Gdrive: https://drive.google.com/drive/folders/1qj5e3j5K3EIRReCnmiji62vs1M0S6NX9?usp=sharing






### Paper
  - [FPHA-Afford: A Domain-Specific Benchmark Dataset for Occluded Object Affordance Estimation in Human-Object-Robot Interaction](https://ieeexplore.ieee.org/document/9190733)  
    <details> <summary>Abstract</summary> 
    In human-object-robot interactions, the recent explosion of standard datasets has offered promising opportunities for deep learning techniques in understanding the functionalities of object parts. But most of existing datasets are only suitable for the applications where objects are non-occluded or isolated during interaction while occlusion is a common challenge in practical object affordance estimation task. In this paper, we attempt to address this issue by introducing a new benchmark dataset named FPHA-Afford that is built upon the popular dataset FPHA. In FPHA-Afford, we adopt egocentric-view to pre-process the videos from FPHA and select part of the frames that contain objects under the strong occlusion of hand. To transfer the domain of FPHA into object affordance estimation task, all of the frames are re-annotated with pixel-level affordance masks. In total, our FPHA-Afford collects 61 videos containing 4.3K frames with 6.55K annotated affordance masks belonging to 9 classes. Some of state-of-the-art semantic segmentation architectures are explored and evaluated over FPHA-Afford. We believe the scale, diversity and novelty of our FPHA-Afford could offer great opportunities to researchers in the computer vision community and beyond. Our dataset and experiment code will be made publicly available on https://github.com/Hussainflr/FPHA-Afford < /details> 
  
  - [Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures](https://github.com/eladrich/latent-nerf)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Code]](https://github.com/eladrich/latent-nerf) 
    <details> <summary>Abstract</summary> 
    Text-guided image generation has progressed rapidly in recent years, inspiring major breakthroughs in text-guided shape generation. Recently, it has been shown that using score distillation, one can successfully text-guide a NeRF model to generate a 3D object. We adapt the score distillation to the publicly available, and computationally efficient, Latent Diffusion Models, which apply the entire diffusion process in a compact latent space of a pretrained autoencoder. As NeRFs operate in image space, a naïve solution for guiding them with latent score distillation would require encoding to the latent space at each guidance step. Instead, we propose to bring the NeRF to the latent space, resulting in a Latent-NeRF. Analyzing our Latent-NeRF, we show that while Text-to-3D models can generate impressive results, they are inherently unconstrained and may lack the ability to guide or enforce a specific 3D structure. To assist and direct the 3D generation, we propose to guide our Latent-NeRF using a Sketch-Shape: an abstract geometry that defines the coarse structure of the desired object. Then, we present means to integrate such a constraint directly into a Latent-NeRF. This unique combination of text and shape guidance allows for increased control over the generation process. We also show that latent score distillation can be successfully applied directly on 3D meshes. This allows for generating high-quality textures on a given geometry. Our experiments validate the power of our different forms of guidance and the efficiency of using latent rendering. < /details> 
  
  - [PaletteNeRF: Palette-based Appearance Editing of Neural Radiance Fields](https://www.timothybrooks.com/instruct-pix2pix/)  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[Code]](https://github.com/timothybrooks/instruct-pix2pix) 
    <details> <summary>Abstract</summary> 
     Recent advances in neural radiance fields have enabled the high-fidelity 3D reconstruction of complex scenes for novel view synthesis. However, it remains underexplored how the appearance of such representations can be efficiently edited while maintaining photorealism. In this work, we present PaletteNeRF, a novel method for photorealistic appearance editing of neural radiance fields (NeRF) based on 3D color decomposition. Our method decomposes the appearance of each 3D point into a linear combination of palette-based bases (i.e., 3D segmentations defined by a group of NeRF-type functions) that are shared across the scene. While our palette-based bases are view-independent, we also predict a view-dependent function to capture the color residual (e.g., specular shading). During training, we jointly optimize the basis functions and the color palettes, and we also introduce novel regularizers to encourage the spatial coherence of the decomposition. Our method allows users to efficiently edit the appearance of the 3D scene by modifying the color palettes. We also extend our framework with compressed semantic features for semantic-aware appearance editing. We demonstrate that our technique is superior to baseline methods both quantitatively and qualitatively for appearance editing of complex real-world scenes. < /details> 



  
