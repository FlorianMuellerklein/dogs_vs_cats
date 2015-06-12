# Dogs vs Cats

VGG style convolution neural network with very leaky ReLU for the kaggle Dogs vs Cats competition. Currently gets 95% on kaggle leaderboards without using outside data and instead realying heavily on data augmentation for generalization.

### Architecture

| Layer Type | Parameters |
| -----------|----------- |
| Input      | size: 168x168, channel: 3 |
| convolution| kernel: 3x3, channel: 32 |
| leaky ReLU | alpha = 0.33 |
| convolution| kernel: 3x3, channel: 32 |
| leaky ReLU | alpha = 0.33 |
| max pool | kernel: 2x2 |
| dropout | 0.1 |
| convolution| kernel: 3x3, channel: 64 |
| leaky ReLU | alpha = 0.33 |
| convolution| kernel: 3x3, channel: 64 |
| leaky ReLU | alpha = 0.33 |
| max pool | kernel: 2x2 |
| dropout | 0.2 |
| convolution| kernel: 3x3, channel: 128 |
| leaky ReLU | alpha = 0.33 |
| convolution| kernel: 3x3, channel: 128 |
| leaky ReLU | alpha = 0.33 |
| max pool | kernel: 2x2 |
| dropout | 0.3 |
| fully connected | units: 2048 |
| leaky ReLU | alpha = 0.33 |
| dropout | 0.5 |
| fully connected | units: 2048 |
| leaky ReLU | alpha = 0.33 |
| dropout | 0.5 |
| softmax | |

### Data augmentation

Images are randomly transformed 'on the fly' while they are being prepared in each batch. The CPU will prepare each batch while the GPU will run the previous batch through the network. 

Random rotations between -30 and 30 degrees.
Random cropping between -24 and 24 pixels in any direction. 
Random zoom between factors of 1 and 1.3. 
Random shearing between -10 and 10 degrees.
Random intensity scaling on RGB channels, independent scaling on each channel.

### To-do

Stream data from SSD instead of holding all images in memory (need to install SSD first).
Try different network archetectures and data pre-processing.
Try intensity scaling method from Krizhevsky, et al 2012.