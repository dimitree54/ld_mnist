# Low dimensional MNIST Dataset

When I provide experiments with custom neural models or alternative training rules, I usually use lightweight datasets such as MNIST, CIFAR or IRIS. But sometimes I am afraid that my experimental models work not well enough not because they are bad, but because of extra difficulties of the toy tasks. For example MNIST has a significant dimensionality reduction (from 784 to 10). Also MNIST has different input-output data structure and distribution (input is dense black and white pixels and output is a sparse one-hot vector).

Looking for a simpler toy dataset for my experimental models I have created a “Low dimensional MNIST Dataset” where 784-pixels images mapped to the 32-dimensional vector using VAE and 10-dimensional labels also mapped to the 32-dimensional vector using VAE. Both of these latent vectors have the same structure and similar distribution (close to standard normal). Now I can test my custom models for the ability to map one normally distributed vector to another.

## Download links:
You can create this dataset on your own with any other latent dimension size, or download my with 32 dimensions from here. VAE which I used for dataset creation you can find here. Using these models you can reconstruct close to original images and labels from your 32-dimensional vectors.


## Dataset statistics.
Both image and class latent vectors have similar distribution close to standard normal. Here is united components hist:

Here is per-component mean and std.

## Image VAE
Here examples of the Image VAE reconstruction abilities 

I am using a very big dropout rate at an image VAE (0.9) to spread information across latent components. Otherwise, almost all the variation can be encoded by 2-d latent vectors. Despite such a big dropout rate some components still do not take part in the decoding process. For example here is visualization of variation at 2 different components of the latent vector. I have chosen such components to show that some affect significantly on the deconstructed image, and some do not.

So my experimental model has to figure out that some input components do not contain a lot of information.

## Class VAE
Class VAE is able to reconstruct the correct class label with the X accuracy.

## Middle layers
As a baseline an exemplar neural network trains this mapping function between latent vectors during both VAE training (it is not necessary to train it in parallel, but it is interesting to observe how its quality evolves as VAE latent vectors change). It uses a separate optimizer, so it does not affect VAE training. The loss for these middle layers is binary_crossentropy of classification of the whole “Image VAE Encoder -> Several of Middle layers -> Class VAE decoder” network. It achieves X train and X val accuracy on that task.

## Benchmark
This dataset is intended to train mapping from the input real vector to the output real vector, so it is a regression task, not classification. So I suppose MSE of X to Y mapping can be used as the main quality metric. As additional metric (not for comparison, just for fun) we can use classification accuracy of “Image VAE Encoder -> Test Model -> Class VAE decoder”, but in that case we need to use original dataset labels (note that samples at the ld_mnist provided in the same order as original order from `tf.keras.datasets.mnist.load_data()`). `eval.evaluate()` function calculates these scores for your model.


To submit your result (that you got on the dataset provided with the download link) email me your score (MSE, reconstructed accuracy), repository with your model and short description.
