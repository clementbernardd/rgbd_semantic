# RGBD semantic segmentation based on Global Convolutional Network in PyTorch


Implementation of the article : RGBD Semantic Segmentation Based on Global ConvolutionalNetwork, 2019 (https://www.researchgate.net/publication/335876739_RGBD_Semantic_Segmentation_Based_on_Global_Convolutional_Network).


I implemented the two dual encoders of the article in PyTorch : FuseNet and GCN.

<p align="center">

<a>
    <img src='images/results.png'  width="900"/>
</a>
</p>
















## Getting Started

### Structure

- `code` : The main directory where all the codes are
- `images` : Images used for the presentation
- `model` : Directory where the model parameters are stored
- `tmp` : Directory where the tensorboard metrics are stored




## Dataset preparation

### NYUv2 Dataset

This dataset contains 1449 RGBD images for 894 different classes.



| Original dataset distribution (truncated) | Dataset used (21 classes) |
|---| --- |
| ![](/images/distribution_classes.png) | ![](/images/distribution_20_classes.png) |

I only kept the 21 classes that were the most present in the dataset


| Original RGB image (480x640x3) | Subsampled RGB image used (240x320x3) |
|---| --- |
| ![](/images/original_rgb.png) | ![](/images/sub_rgb.png) |

I subsampled the images by 2

| Original Labels (480x640x1) with 894 classes  | Subsampled labels used (240x320x1) with 21 classes  |
|---| --- |
| ![](/images/all_labels.png) | ![](/images/20_labels.png) |


## Models

The models of the articles are two dual encoder-decoder where the depth is added during the encoded phase.

### FuseNet

The encoder for the RGB images has the same architecture as the VGG16-Net.

The architecture is the following :

| FuseNet architecture |
|---|
| ![](/images/fusenet.png) |



| Prediction with FuseNet on training set | Label |
|---| --- |
| ![](/images/fusenet_train.png) | ![](/images/fusenet_train_label.png) |

| Prediction with FuseNet on testing set | Label |
|---| --- |
| ![](/images/fusenet_test.png) | ![](/images/fusenet_test_label.png) |


### GCN

It has almost the same architecture as FuseNet. It adds extra layers of global convolutional network (convolution of size kxk).
Here the convolution added is 5x5.

The architecture is the following :

| GCN architecture |
|---|
| ![](/images/gcn.png) | 



| Prediction with GCN on training set | Label |
|---| --- |
| ![](/images/gcn_train.png) | ![](/images/gcn_train_label.png) |

| Prediction with GCN on testing set | Label |
|---| --- |
| ![](/images/gcn_test.png) | ![](/images/gcn_test_label.png) |



### Results

Here is a summary of the final scores :

| Scores |
|---|
| ![](/images/histogram.png) |




<table align="center">
<tr>
<td><b> Models <td><b> FuseNet without transfer learning <td><b> FuseNet with transfer learning (VGG16-Net) <td><b> GCN with transfer learning (VGG16-Net) <td> <b> FuseNet of the article <td> <b> GCN of the article
<tr>
<td> Mean accuracy <td> 25.37 <td> 33.51 <td> 39.24 <td> 46.42 <td> 48.49
<tr>
<td> Mean IoU <td> 16.87 <td> 20.27 <td> 23.50 <td> 35.48 <td> 36.94
<tr>
<td> Pixel accuracy <td> 46.63 <td> 51.30 <td> 55.62 <td> 68.76 <td> 69.11
<tr>
</table>

Final summary of the results :

| Summary |
|---|
| ![](/images/results.png) |
