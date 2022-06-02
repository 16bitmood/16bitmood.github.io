---
layout: post.njk
title: Mapping Images to Images
date: 2022-06-02
timeToRead: ~ 7
tags: blog
---

<div>
<img style='max-height: 20em' src='/img/pix2pix-example.png'> </img>
<div class='img-caption'>
    <span> Actual $X$ </span>
    <span> Actual $Y$ </span>
    <span> AI Generated $X \rightarrow Y$ </span>
</div>
</div>

Many problems in image processing, graphics, and vision involve transforming an
input image into a corresponding output image. There exist a wide variety of
domain specific solutions for many problems&mdash;Colorization,
Edge Detection,
Segmentation,
Neural Style Transfer, etc.&mdash;even though the goal is always the
same: mapping pixels to pixels.


One obvious question we can ask is: does there exist a general-purpose solution
to this problem?  That is, given two *paired* domains of images $X$ and $Y$,
can we always find a function 
$G : X \rightarrow Y$?

<img style='max-height: 20em' src='/img/gan-visual.png'> </img>
<div class='img-caption'>
    <span>Generative Adversarial Networks</span>
</div>

It turns out, yes! We can always find such a function $G$. To build $G$, we
will utilize a class of generative machine learning models known as 
[Generative Adversarial Networks](/posts/gan).

# Can We Use a Simple CNN?
Before considering complicated solutions, first question we should ask is: do we
need GANs at all, or will a simple CNN suffice?  There exist several generative
CNN models&mdash;Encoder-Decoder, U-Net, etc.&mdash;so why don't we
use them?

The problem is as follows: CNNs learn to minimize a loss function, and although
the learning process is automatic, we still need to carefully design the loss
function. Which means, **we need to tell the CNN what 
we want to minimize**!

But if we take a naive approach and tell the CNN to
minimize the euclidean distance between the input image
and output image, we get *blurry* images.
This happens because it is much "safer" for the L2 loss to predict the
mean of the distribution, as that minimizes the mean pixel-wise error.

Rather than specifying a carefully crafted model and loss for each
Image-to-Image problem. It would be better if we could specify
some goal like: **"make the output indistinguishable from reality"**.

While this may seem like a vague target, this is exactly what 
[Generative Adversarial Networks](/posts/gan) claim to do!

# Enter Pix2Pix

<img style='max-height: 20em' src='/img/pix2pix-cgan.png'> </img>
<div class='img-caption'>
<span> cGAN for Pix2Pix </span>
</div>


The main idea behind the [Pix2Pix](https://arxiv.org/abs/1611.07004) paper is to use
a Conditional [Generative Adversarial Network](/posts/gan) to learn the loss function,
rather than hand-coding a loss function&mdash;such as Euclidean distance&mdash;between
target and input images. 

## Objective

<img style='max-height: 20em; width: auto;' src='/img/pix2pix-paired.png'> </img>
<div class='img-caption'>
    <span>Supervised Image-to-Image Translation</span>
</div>

The goal is to generate a target image, which is based on some source image. 
As we don't want our model to generate random images, we use a 
variant of GANs called Conditional GANs.



As the name suggests, Conditional GANs take a *conditional variable* to both the generator $G$, and 
the discriminator $D$. For example, if we have a training pair
$(c, y)$, where we want to generate the item $y$ based on the
value of $c$, we pass $c$ to both $G$ and $D$. 

The loss function then becomes:

<div>
$$
    \mathcal{L}_{\text{cGAN}}(G,D) = 
        \mathbb{E}_{x, c} \log D(c, x)
        +
        \mathbb{E}_{z, c} \log (1 - D(c, G(c, z) ))
$$
</div>

In their experiments they found mixing both the adversarial loss
and a more traditional loss&mdash;such as L1 loss&mdash;was helpful for training
the GAN. The loss function for the discriminator remains unchanged
but the generator now not only has to fool the discriminator
but also minimize the L1 loss between the predicted and target
image.
<div>
$$
    \mathcal{L}_{\text{L1}}(G, D) = \mathbb{E}_{y, c, z}
    \left\|  y - G(c, z)  \right\|
$$
</div>

The final objective becomes for the generator becomes:
<div>
$$
    G^* = \arg\min_G \max_D
    \mathcal{L}_{\text{cGAN}}(G,D)
    + \lambda \mathcal{L}_{\text{L1}}(G, D)
$$
</div>

Where $\lambda$ is a hyperparameter for our model.

**Note:** For training the generator, instead of providing random noise as a direct
input, they provide noise in the form of dropout.

## Discriminator Architecture


<img style='height: 20em; width: auto;' src='/img/pix2pix-patchgan.png'> </img>

<div class='img-caption'>
<span> Visualizing PatchGAN </span>
</div>


Since we are already adding a traditional loss such as L1(or L2)
which&mdash;although produce blurry results when used alone&mdash;are very
useful for enforcing correctness at the low frequency. This is the main idea
behind the **PatchGAN** discriminator.

**PatchGAN** architecture restricts the discriminator to model
only high frequency structure in the generated image. To model this,
it is sufficient to restrict the attention to local image patches.


Therefore, rather than generating a single probability in range $[0, 1]$, the
discriminator generates a $k \times k$ matrix of values
in range $[0, 1]$ where each element in the matrix corresponds to a local
patch(the *receptive field*) of the image.

One advantage of **PatchGAN** is that the same architecture
can extend to images of higher resolution with same number of parameters, 
and it would generate a similar, but larger, $k' \times k'$ matrix.

Inspired by **DCGAN**, the discriminator is implemented as a
Deep CNN, build up of repetitions of modules of form: Convolution -
BatchNorm - ReLU, where we down-sample
until we reach the required $k \times k$ matrix, which we then pass through
the logistic function to get the elements in range $[0, 1]$.


## Generator Architecture

<img style='max-height: 20em' src='/img/pix2pix-autoencoder.png'> </img>
<div class='img-caption'>
<span> Encoder-Decoder Generator Architecture </span>
</div>

One of the obvious choices for a generator architecture for Image-to-Image
translation, is the encoder-decoder network. 


In this kind of network, 
the input image is passed through a series of layers which down-sample
the resolution but increases the number of channels until we reach
an image of size $1 \times 1$, after that the process is reversed 
until we reach the resolution of the original image.


Here also we use modules of the form: Convolution - BatchNorm - LeakyReLU
for encoder, for decoder we use: TransposeConvolution - BatchNorm - ReLU.


For many interesting image-translation problems, there is a lot of
low-level information shared between the input and output image.
For example, in the case of colorization, the input and output
share the location of edges. To preserve this information,
we add skip connections to create a "U-net" like architecture.


<img style='max-height: 20em' src='/img/pix2pix-unet.png'> </img>
<div class='img-caption'>
<span> U-Net Generator Architecture </span>
</div>

This network is similar to Encoder-Decoder network,
but we add some extra skip-connections between layer 
$i$ in the encoder and layer $n-i$ in the decoder.

# Experiments

<video autoplay muted loop>
    <source src="/img/pix2pix-experiments2.mp4" type="video/mp4">
</video>

<div class='img-caption'>
    <span> Results of cGAN changing per epoch </span>
</div>


To demonstrate the versatility of the *Pix2Pix* approach, I applied the model
on three different datasets with seemingly very different problem setting.

The datasets used are: 
[Satellite2Map](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/), 
[Edges2Shoes](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/), and
[Anime Sketch Colorization](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair).

I used the *Adam* Optimizer with hyperparameters 
$\beta_1 = 0.5$ and $\beta_2 = 0.999$. Learning rate was set to $0.0002$, and 
$\lambda$ used was $100$.

## Satellite $\rightarrow$ Map

This dataset consists of paired images of aerial photography and their
corresponding map view scraped using google maps. There are $1096$ training
images of size $256 \times 256$. The model was trained for $150$ epochs with
batch size $1$.

![Image](/img/pix2pix-map.png "Example")

<div class='img-caption'>
<span> Actual $X$ </span>
<span> Actual $Y$ </span>
<span> AI Generated $X \rightarrow Y$ </span>
</div>


## Edges $\rightarrow$ Shoe

This dataset consists of paired images of
sketches of shoes and the corresponding actual image
(created using edge detection). There are $50$k training images of size 
$256 \times 256$. The model was trained for $30$ epochs with
batch size $1$.

![Image](/img/pix2pix-shoe.png "Example")

<div class='img-caption'>
<span> Actual $X$ </span>
<span> Actual $Y$ </span>
<span> AI Generated $X \rightarrow Y$ </span>
</div>


## Sketch $\rightarrow$ Colored
This dataset consists of paired images of anime line sketch images and 
the corresponding colored image. There are $14$k training images of size 
$256 \times 256$. The model was trained for $75$ epochs with
batch size $1$.


![Image](/img/pix2pix-anime.png "Example")

<div class='img-caption'>
<span> Actual $X$ </span>
<span> Actual $Y$ </span>
<span> AI Generated $X \rightarrow Y$ </span>
</div>

# References
1. Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David
    Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014.
    “Generative Adversarial Nets.” *Advances in Neural Information Processing
    Systems* 27.
2. Isola, Phillip, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. 2017.
    “Image-to-Image Translation with Conditional Adversarial Networks.” In
    *Proceedings of the Ieee Conference on Computer Vision and Pattern
    Recognition*, 1125–34.