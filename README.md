# Dogs vs Cats

VGG style convolution neural network with very leaky ReLU for the kaggle Dogs vs Cats competition. Currently gets 96.6% on kaggle leaderboards without using outside data and instead relying heavily on data augmentation for generalization. Small amount of fine tuning (finishign training with a small number of iterations with very low learning rate and no data augmentation).

### Architecture

| Layer Type | Parameters |
| -----------|----------- |
| Input      | size: 168x168, channel: 3 |
| convolution| kernel: 3x3, channel: 32 |
| leaky ReLU | alpha = 0.2 |
| convolution| kernel: 3x3, channel: 32 |
| leaky ReLU | alpha = 0.2 |
| max pool | kernel: 2x2 |
| dropout | 0.1 |
| convolution| kernel: 3x3, channel: 64 |
| leaky ReLU | alpha = 0.2 |
| convolution| kernel: 3x3, channel: 64 |
| leaky ReLU | alpha = 0.2 |
| max pool | kernel: 2x2 |
| dropout | 0.2 |
| convolution| kernel: 3x3, channel: 128 |
| leaky ReLU | alpha = 0.2 |
| convolution| kernel: 3x3, channel: 128 |
| leaky ReLU | alpha = 0.2 |
| convolution| kernel: 3x3, channel: 128 |
| leaky ReLU | alpha = 0.2 |
| max pool | kernel: 2x2 |
| dropout | 0.3 |
| fully connected | units: 1024 |
| leaky ReLU | alpha = 0.2 |
| dropout | 0.5 |
| fully connected | units: 1024 |
| leaky ReLU | alpha = 0.2 |
| dropout | 0.5 |
| softmax | |

### Data augmentation

Images are randomly transformed 'on the fly' while they are being prepared in each batch. The CPU will prepare each batch while the GPU will run the previous batch through the network. 

* Random rotations between -30 and 30 degrees.
* Random cropping between -24 and 24 pixels in any direction. 
* Random zoom between factors of 1 and 1.3. 
* Random shearing between -10 and 10 degrees.
* Random intensity scaling on RGB channels, independent scaling on each channel.

![Imgur](http://i.imgur.com/rW0a8Yx.png) ![Imgur](http://i.imgur.com/Xg6zouG.gif)

### To-do

Stream data from SSD instead of holding all images in memory (need to install SSD first).
Try different network archetectures and data pre-processing.
Try intensity scaling method from Krizhevsky, et al 2012.

### References

* Karen Simonyan, Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", [link](http://arxiv.org/abs/1409.1556)
* Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", [link](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* Sander Dieleman, "Classifying plankton with deep neural networks", [link](http://benanne.github.io/2015/03/17/plankton.html)