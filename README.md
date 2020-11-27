# RGBD semantic segmentation based on Global Convolutional Network in PyTorch


Implementation of the article : RGBD Semantic Segmentation Based on Global ConvolutionalNetwork, 2019 (https://www.researchgate.net/publication/335876739_RGBD_Semantic_Segmentation_Based_on_Global_Convolutional_Network).


I implemented the two dual encoders of the article in PyTorch : FuseNet and GCN.

<p align="center">

<a>
    <img src='images/results.png'  width="900"/>
</a>
</p>
















## Getting Started



## Dataset preparation

### NYUv2 Dataset

This dataset contains 1449 RGBD images for 894 different classes.

| Original dataset distribution (truncated) | Dataset used (21 classes) |
|---| --- |
| ![](/images/distribution_classes.png) | ![](/images/distribution_20_classes.png) |

| Original RGB image ( 480 x 640 x 3 ) | Subsampled RGB image used (240 x 320 x 3) |
|---| --- |
| ![](/images/original_rgb.png) | ![](/images/sub_rgb.png) |

| Original Labels ( 480 x 640 x 1 ) with 894 classes  | Subsampled labels used (240 x 320 x 1) with 21 classes  |
|---| --- |
| ![](/images/original_labels.png) | ![](/images/sub_labels.png) |





## Models



### FuseNet


### GCN



### Results
