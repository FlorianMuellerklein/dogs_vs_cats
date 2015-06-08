# Dogs vs Cats

VGG style convolution neural network for the kaggle Dogs vs Cats competition. Currently gets 94.5% on kaggle leaderboards without using outside data. 

### Architecture

The input was 192x192 rgb images. 
6 convolution layers with filter size 3x3 and ReLU activations. Max pooling layers after every other convolution layer.
2 hidden layers with dropout. Softmax output.

### Data augmentation

Images are randomly transformed 'on the fly' while they are being prepared in each batch. The CPU will prepare each batch while the GPU will run the previous batch through the network. 

Random rotations between -30 and 30 degrees.
Random translations between -24 and 24 pixels in any direction. 
Random zoom between factors of 1 and 1.3. 
Random shearing between -10 and 10 degrees. 