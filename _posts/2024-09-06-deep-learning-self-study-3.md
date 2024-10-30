---
layout: post
title: Deep Learning - Basics of Computer Vision (ConvNets)
date: 2024-09-06 18:49:00
description: "Topics covered: Convolutional Neural Network Foundations, Case Studies, Object Detection, Special applications (Face Recognition and Neural Style Transfer)."
tags: machine-learning deep-learning computer-vision
categories: deep-learning
toc:
  sidebar: left
bibliography: 2024-06-30-dl-1.bib
related_publications: true
pretty_table: true
---

Simple Artificial Neural Networks (ANNs), though powerful by themselves face a hard limit to how effective they can be when working with specialized domains such as computer vision, speech recognition and so on. Such implementations require completely different architectures, concepts and implementations, and, as has been discussed before, intuitions for these domains do not always cross pollinate.

This blog will cover one such domain that has been of great interest to me for the longest time: **Computer Vision**. A lot of the content of this blog is based off papers written by the original authors with the main structure of the blog being highly inspired by the [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning) portion of the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by Andrew Ng. The images in this blog were either made by hand, in [draw.io](https://app.diagrams.net/), taken from their respective research papers, or were generated through python. Without wasting any more time, let's begin.

<hr>

# What is Computer Vision?

Computer Vision (CV) is one of the most intuitive and rapidly advancing fields of deep learning in the modern day. The basic idea of computer vision is to allow devices to be able to observe and identify images and/or aspects of images and videos, trying to mimic how a human's ocular nervous system would do the same.

As groundbreaking as it is, it would surprise you to know that computer vision research is not as novel as one would think it to be. The ideas have been around since the early 60s, with the first foray into this field being through neurophysical {% cite Hoffmann1984 %} experimentation son cats. It is even more intriguing to know that the ideas that worked towards bringing about the discussion of AI stems from the ideas of mechanical autonomy and intelligent machines almost 3000 years ago. Homer's Odyssey is a prime example of literature at the time that brushes upon the idea of intelligent machines {% cite Lively2020 %}.

Finding use in several domains such as medical imaging, autonomous vehicles, quality control and augmented reality, the rapid advancements in computer vision have pushed us towards new horizons that we would never have fathomed before. Some popular implementations of computer vision are as follows:
- **Image Recognition**: Image recognition is one of the most common computer vision problems. This classification problem involves training specifically designed neural network models on several examples of labelled image inputs, learning different aspects of it, with the end goal of determining the identity of new and unknown examples.
- **Object Detection**: Image recognition and object detection are similar techniques but differ completely in their applications. While image recognition identifies an object in an image, object detection is used to detect and locate instances of objects in sample images or videos. Here, the models create bounding boxes at the position and boundaries of the objects we want to detect. Image recognition is often used alongside object detection in several applications.
- **Image Segmentation**: Image segmentation can be seen as a more refined form of object detection and image classification. Here, the models process the visual data in the image at a pixel level. Classical image segmentation methods take into account the heuristics of the pixels in the image such as the color, position, and intensity. These are used in neural networks to generate more sophisticated results. The outputs of segmentation are segmentation masks which act as pixel-to-pixel boundaries around the objects' shape for each class of object being identified.

There are several other applications of computer vision in addition to the above, with neural style transfer, autonomous vehicles, augmented reality and facial recognition being just a few of their implementations.

# Convolutional Neural Networks

## Foundations of Convolutional Neural Networks

The main issue that computer vision problems pose when using normal deep neural networks is the fact that, as the dimensions of the images being used increases, the number of input features we will have to deal with becomes proportionally larger.

As a result it is difficult to get enough data to prevent overfitting. In addition, this will also lead to a large computational requirement when training the network.

The solution for this is the **convolution operation**. This is the most building block of a significant number of computer vision models, also known as **convolutional neural networks (CNNs)**.

## The Convolution Operation

The best explanations of the convolution operation in signal processing that I have seen are from this video by [3Blue1Brown](https://www.youtube.com/@3blue1brown).

<div class="row mt-1">
    <div class="col-sm mt-1 mt-md-0">
        <iframe width="560" height="315" src="https://www.youtube.com/embed/KuXjwB4LzSA?si=ozem701wKPFMVaac" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
    </div>
</div>

<br>

There are several ways of picturing what a convolution is and how it works. It even ties into moving averages which was discussed in the [previous blog]({{site.baseurl}}{% link _posts/2024-07-22-deep-learning-self-study-2.md %}).

The official formulaic definition for the convolution of two different functions is as follows

$$(f*g)(t) = \int_{-\infty}^{\infty} f(x)g(t-x)$$

However this formula tends to scare off most people who do not know what is going on exactly. This is where different intuitions on this help.

We will be looking at convolutions from the standpoint of computer vision, where they play the role of filtering images for various uses such as smoothing, sharpening, denoising or detecting edges. This is done so with the help of a convolution **filter** or **kernel** which moves across the pixels in the image, multiplying the values and adding them up, giving us a different version of the same image.

## An intuition: Vertical Edge Detection

As has been mentioned before, the good way to understand how convolution works is in the field of computer vision, through the example of a **filter**.

In this case we are making use of an edge detector, specifically a vertical edge detector, which we will look at after this section.

Consider a $$6\times 6$$ greyscale image, which is effectively a 2 dimensional matrix of dimensions $$(6,6,1)$$ as shown below

$$
\begin{bmatrix}
3 & 0 & 1 & 2 & 7 & 4 \\
1 & 5 & 8 & 9 & 3 & 1 \\
2 & 7 & 2 & 5 & 1 & 3 \\
0 & 1 & 3 & 1 & 7 & 8 \\
4 & 2 & 1 & 6 & 2 & 8 \\
2 & 4 & 5 & 2 & 3 & 9 \\
\end{bmatrix}
$$

For this image we, will be applying a vertical edge detection filter which is of the form

$$
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
$$

to it. The convolving setup is as follows.

$$
\begin{bmatrix}
3 & 0 & 1 & 2 & 7 & 4 \\
1 & 5 & 8 & 9 & 3 & 1 \\
2 & 7 & 2 & 5 & 1 & 3 \\
0 & 1 & 3 & 1 & 7 & 8 \\
4 & 2 & 1 & 6 & 2 & 8 \\
2 & 4 & 5 & 2 & 3 & 9 \\
\end{bmatrix}
*
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
= 
\begin{bmatrix}
* & * & * & * \\
* & * & * & * \\
* & * & * & * \\
* & * & * & * \\
\end{bmatrix}
$$

where the output is a $$4 \times 4$$ array or image.

Now, we want to apply the convolution to the image with the given filter, the computation process of which is as follows:

We first perform the convolution on the upper left corner of the image by pasting the given $$3 \times 3$$ filter on the upper left $$3 \times 3$$ region of the original input image. We then perform the element-wise product and add up all the $$9$$ resulting numbers to get the first pixel value for our convoluted result.

$$
\begin{bmatrix}
\colorbox{grey}{3^{1}} & \colorbox{grey}{0^{0}} & \colorbox{grey}{1^{-1}} & 2 & 7 & 4 \\
\colorbox{grey}{1^{1}} & \colorbox{grey}{5^{0}} & \colorbox{grey}{8^{-1}} & 9 & 3 & 1 \\
\colorbox{grey}{2^{1}} & \colorbox{grey}{7^{0}} & \colorbox{grey}{2^{-1}} & 5 & 1 & 3 \\
0 & 1 & 3 & 1 & 7 & 8 \\
4 & 2 & 1 & 6 & 2 & 8 \\
2 & 4 & 5 & 2 & 3 & 9 \\
\end{bmatrix}
*
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
= 
\begin{bmatrix}
\colorbox{grey}{x} & * & * & * \\
* & * & * & * \\
* & * & * & * \\
* & * & * & * \\
\end{bmatrix}
$$

$$
\begin{align*}
\Rightarrow x = \;\; & (3 \times 1) + (0 \times 0) + (1 \times (-1))\\
& + (1 \times 1) + (5 \times 0) + (8 \times (-1))\\
& + (2 \times 1) + (7 \times 0) + (2 \times (-1))
\end{align*}
$$

Similarly, for the second, third and fourth elements, the same process is repeated:

$$
\begin{bmatrix}
3 & \colorbox{grey}{0^{1}} & \colorbox{grey}{1^{0}} & \colorbox{grey}{2^{-1}} & 7 & 4 \\
1 & \colorbox{grey}{5^{1}} & \colorbox{grey}{8^{0}} & \colorbox{grey}{9^{-1}} & 3 & 1 \\
2 & \colorbox{grey}{7^{1}} & \colorbox{grey}{2^{0}} & \colorbox{grey}{5^{-1}} & 1 & 3 \\
0 & 1 & 3 & 1 & 7 & 8 \\
4 & 2 & 1 & 6 & 2 & 8 \\
2 & 4 & 5 & 2 & 3 & 9 \\
\end{bmatrix}
*
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
= 
\begin{bmatrix}
-5 & \colorbox{grey}{-4} & * & * \\
* & * & * & * \\
* & * & * & * \\
* & * & * & * \\
\end{bmatrix}
$$

$$
\begin{bmatrix}
3 & 0 & \colorbox{grey}{1^{1}} & \colorbox{grey}{2^{0}} & \colorbox{grey}{7^{-1}} & 4 \\
1 & 5 & \colorbox{grey}{8^{1}} & \colorbox{grey}{9^{0}} & \colorbox{grey}{3^{-1}} & 1 \\
2 & 7 & \colorbox{grey}{2^{1}} & \colorbox{grey}{5^{0}} & \colorbox{grey}{1^{-1}} & 3 \\
0 & 1 & 3 & 1 & 7 & 8 \\
4 & 2 & 1 & 6 & 2 & 8 \\
2 & 4 & 5 & 2 & 3 & 9 \\
\end{bmatrix}
*
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
= 
\begin{bmatrix}
-5 & -4 & \colorbox{grey}{0} & * \\
* & * & * & * \\
* & * & * & * \\
* & * & * & * \\
\end{bmatrix}
$$

$$
\begin{bmatrix}
3 & 0 & 1 & \colorbox{grey}{2^{1}} & \colorbox{grey}{7^{0}} & \colorbox{grey}{4^{-1}} \\
1 & 5 & 8 & \colorbox{grey}{9^{1}} & \colorbox{grey}{3^{0}} & \colorbox{grey}{1^{-1}} \\
2 & 7 & 2 & \colorbox{grey}{5^{1}} & \colorbox{grey}{1^{0}} & \colorbox{grey}{3^{-1}} \\
0 & 1 & 3 & 1 & 7 & 8 \\
4 & 2 & 1 & 6 & 2 & 8 \\
2 & 4 & 5 & 2 & 3 & 9 \\
\end{bmatrix}
*
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
= 
\begin{bmatrix}
-5 & -4 & 0 & \colorbox{grey}{8} \\
* & * & * & * \\
* & * & * & * \\
* & * & * & * \\
\end{bmatrix}
$$

The same steps are repeated for the subsequent rows, giving us the following result

$$
\begin{bmatrix}
3 & 0 & 1 & 2 & 7 & 4 \\
1 & 5 & 8 & 9 & 3 & 1 \\
2 & 7 & 2 & 5 & 1 & 3 \\
0 & 1 & 3 & 1 & 7 & 8 \\
4 & 2 & 1 & 6 & 2 & 8 \\
2 & 4 & 5 & 2 & 3 & 9 \\
\end{bmatrix}
*
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
= 
\begin{bmatrix}
-5 & -4 & 0 & 8 \\
-10 & -2 & 2 & 3 \\
0 & -2 & -4 & -7 \\
-3 & -2 & -3 & -16 \\
\end{bmatrix}
$$

With this, we have performed a single convolution on our given image, giving us a result that detects the vertical edges in our image.

## Edge Detectors

This result obtained in the previous section is known as a Vertical Edge Detector, which is one of the many types of edge detection kernels that deep learning models inadvertently end up learning in the lower layers of their networks.

These kernels, as the name suggests, find any parts of our image that may have vertical, or horizontal edges.

Edge detection works by trying to find regions in an image where there is a sharp change in intensity. A higher value would indicate a steep change while a lower value indicates a shallow change.

Consider the given example below where we have a lighter region with the value of $$10$$ at each pixel and a darker region at a value of $$0$$.

$$
\begin{bmatrix}
10 & 10 & 10 & 0 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 & 0 \\
\end{bmatrix}
$$

<div class="row justify-content-center mt-3">
    <div class="col-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/beforeConv.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The array above would look something like this image.
</div>

The filter we are using here is of the form

$$
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
$$

<div class="row justify-content-center mt-3">
    <div class="col-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/filterConv.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The filter we are using looks something like this.
</div>

Upon performing the convolution operation we obtain the following result.

$$
\begin{bmatrix}
10 & 10 & 10 & 0 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 & 0 \\
\end{bmatrix}
*
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
=
\begin{bmatrix}
0 & 30 & 30 & 0 \\
0 & 30 & 30 & 0 \\
0 & 30 & 30 & 0 \\
0 & 30 & 30 & 0 \\
\end{bmatrix}
$$

Which looks something like the below image

<div class="row justify-content-center mt-3">
    <div class="col-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/afterConv.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

In this case, when the filter is in a region where all the pixel colors are the same, for example the upper left or upper right most sections of the given image example, the vertical edge detecting convolution gives us a zero on the output as the calculations made on these pixels cancel themselves out.

Towards the center of the image where there is a change in the color values of the pixels in the image, the filter outputs a higher value due to the difference in color intensities between the left-hand and right-hand sides.

As a result, this emphasizes upon the change in intensity, giving lesser importance parts of the image where the color intensities of the pixels are either the same or in the similar range.

For an image with the intensities flipped, with the darker side being on the left, the result would be something like the following.

$$
\begin{bmatrix}
0 & 0 & 0 & 10 & 10 & 10 \\
0 & 0 & 0 & 10 & 10 & 10 \\
0 & 0 & 0 & 10 & 10 & 10 \\
0 & 0 & 0 & 10 & 10 & 10 \\
0 & 0 & 0 & 10 & 10 & 10 \\
0 & 0 & 0 & 10 & 10 & 10 \\
\end{bmatrix}
*
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
=
\begin{bmatrix}
0 & -30 & -30 & 0 \\
0 & -30 & -30 & 0 \\
0 & -30 & -30 & 0 \\
0 & -30 & -30 & 0 \\
\end{bmatrix}
$$

<div class="row justify-content-center mt-3">
    <div class="col-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/afterConvInvert.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Examples of Edge Detection

- **Prewitt Operator**: This family of edge detectors is used to detect vertical and horizontal edges in images. It does so by placing an emphasis on the pixels towards a given side of the mas, either the left or right hand side for the vertical filter, and the top or bottom for a horizontal filter.

$$
\text{Vertical Filter} = \begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
$$

$$
\text{Horizontal Filter} = \begin{bmatrix}
1 & 1 & 1 \\
0 & 0 & 0 \\
-1 & -1 & -1 \\
\end{bmatrix}
$$

- **Sobel Operator**: This is one of the most commonly used family of filters in edge detection. Here, the filter places an empahsis on pixels close to the center of the mask as opposed to the Prewitt filters, hence making it a bit more robust.

$$
\text{Vertical Sobel Filter} = \begin{bmatrix}
1 & 0 & -1 \\
2 & 0 & -2 \\
1 & 0 & -1 \\
\end{bmatrix}
$$

$$
\text{Horizontal Sobel Filter} = \begin{bmatrix}
1 & 2 & 1 \\
0 & 0 & 0 \\
-1 & -2 & -1 \\
\end{bmatrix}
$$

## Implementing Vertical and Horizontal Edge Filters

We will now look at an implementation of vertical and horizontal edge detectors in Python to see how they work.

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/filter_demonstrations.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/filter_demonstrations.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

<br>

## A Note on Edge Detection in Neural Networks

In deep learning, it is not required for us to handpick the values of the weights used in the filters acting on the input images. In most cases, these values are treated as parameters which are automatically learnt through the backpropagation process.

In many cases the filters learned by the networks by themselves are much more robust than any predefined filters that we may have.

## Padding

When building neural networks, one of the modifications we have to make to our images is called **padding**.

When performing convolutions on our input images, the dimensions of the output images are reduced with every layer of the neural network.

For example, a $$6 \times 6$$ image being convolved with a $$3 \times 3$$ filter gives us a $$4 \times 4$$ output image. 

Extrapolating this operation to different examples, for a $$n \times n$$ image being convolved with a $$f \times f$$ filter, the output would have the following dimensions.

$$\boxed{(n-f+1) \times (n-f+1)}$$

The downsides here are:
- Every convolution operation will lead to the image dimensions shrinking
- The pixels at the corners of the image are not given as much of an importance as a pixel in the middle as the kernel only acts on it once. As a result, the corner pixels contribute much less to the output.

To combat this, padding is implemented before each convolutional operation.

The convention followed is to pad the image with layers of pixels $$p$$ with the value of $$0$$ around the image, changing its dimensions from $$n \times n$$ to $$ (n + 2p) \times (n + 2p)$$. 

The output resulting from convolutions being performed on this image will be

$$\boxed{(n+2p-f+1) \times (n+2p-f+1)}$$

Increasing the number of layers when padding the image ensures that the corner pixels are utilized more in the outputs.

## Valid and Same Convolutions

A **valid** convolution is one where no padding has been done to the input images

$$(n \times n) * (f \times f) \rightarrow (n-f+1) \times (n-f+1)$$

A **same** convolution means that the image has been padded such that the size of the output is the same as the size of the input

$$
\begin{align*}
& (n+2p-f+1) \times (n+2p-f+1) = (n \times n) \\
\Rightarrow & p = \frac{f-1}{2} 
\end{align*}
$$

**By convention, the value of $$f$$ is always odd**

An even filter size would require uneven padding. If $$f$$ is odd, the padding region from the convolution will be more natural.

When using odd dimensions of filters, this gives us a central pixel, which makes it easier to talk about its position.

## Strided Convolutions

Strided convolutions are another building block of convolutional neural networks. The mechanism for the same will be discussed below.

Consider the given $$7 \times 7$$ image being convolved with a $$3 \times 3$$ filter with a stride of $$2$$

$$
\begin{bmatrix}
2 & 3 & 7 & 4 & 6 & 2 & 9 \\
6 & 6 & 9 & 8 & 7 & 4 & 3 \\
3 & 4 & 8 & 3 & 8 & 9 & 7 \\
7 & 8 & 3 & 6 & 6 & 3 & 4 \\
4 & 2 & 1 & 8 & 3 & 4 & 6 \\
3 & 2 & 4 & 1 & 9 & 8 & 3 \\
0 & 1 & 3 & 9 & 2 & 1 & 4
\end{bmatrix}
*
\begin{bmatrix}
3 & 4 & 4 \\
1 & 0 & 2 \\
-1 & 0 & 3
\end{bmatrix}
$$

The first step here is done in the same way as we would normally do

$$
\begin{bmatrix}
\colorbox{grey}{2^{3}} & \colorbox{grey}{3^{4}} & \colorbox{grey}{7^{4}} & 4 & 6 & 2 & 9 \\
\colorbox{grey}{6^{1}} & \colorbox{grey}{6^{0}} & \colorbox{grey}{9^{2}} & 8 & 7 & 4 & 3 \\
\colorbox{grey}{3^{-1}} & \colorbox{grey}{4^{0}} & \colorbox{grey}{8^{3}} & 3 & 8 & 9 & 7 \\
7 & 8 & 3 & 6 & 6 & 3 & 4 \\
4 & 2 & 1 & 8 & 3 & 4 & 6 \\
3 & 2 & 4 & 1 & 9 & 8 & 3 \\
0 & 1 & 3 & 9 & 2 & 1 & 4
\end{bmatrix}
*
\begin{bmatrix}
3 & 4 & 4 \\
1 & 0 & 2 \\
-1 & 0 & 3
\end{bmatrix}
=
\begin{bmatrix}
\colorbox{grey}{91} & * & * \\
* & * & * \\
* & * & * \\
\end{bmatrix}
$$

From the second step, instead of jumping to the next pixel, we move over by the number of steps mentioned by the stride, which in this case is $$2$$, and perform the next calculation.

$$
\begin{bmatrix}
2 & 3 & \colorbox{grey}{7^{3}} & \colorbox{grey}{4^{4}} & \colorbox{grey}{6^{4}} & 2 & 9 \\
6 & 6 & \colorbox{grey}{9^{1}} & \colorbox{grey}{8^{0}} & \colorbox{grey}{7^{2}} & 4 & 3 \\
3 & 4 & \colorbox{grey}{8^{-1}} & \colorbox{grey}{3^{0}} & \colorbox{grey}{8^{3}} & 9 & 7 \\
7 & 8 & 3 & 6 & 6 & 3 & 4 \\
4 & 2 & 1 & 8 & 3 & 4 & 6 \\
3 & 2 & 4 & 1 & 9 & 8 & 3 \\
0 & 1 & 3 & 9 & 2 & 1 & 4
\end{bmatrix}
*
\begin{bmatrix}
3 & 4 & 4 \\
1 & 0 & 2 \\
-1 & 0 & 3
\end{bmatrix}
=
\begin{bmatrix}
91 & \colorbox{grey}{100} & * \\
* & * & * \\
* & * & * \\
\end{bmatrix}
$$

The same is repeated for the subsequent operation

$$
\begin{bmatrix}
2 & 3 & 7 & 4 & \colorbox{grey}{6^{3}} & \colorbox{grey}{2^{4}} & \colorbox{grey}{9^{4}} \\
6 & 6 & 9 & 8 & \colorbox{grey}{7^{1}} & \colorbox{grey}{4^{0}} & \colorbox{grey}{3^{2}} \\
3 & 4 & 8 & 3 & \colorbox{grey}{8^{-1}} & \colorbox{grey}{9^{0}} & \colorbox{grey}{7^{3}} \\
7 & 8 & 3 & 6 & 6 & 3 & 4 \\
4 & 2 & 1 & 8 & 3 & 4 & 6 \\
3 & 2 & 4 & 1 & 9 & 8 & 3 \\
0 & 1 & 3 & 9 & 2 & 1 & 4
\end{bmatrix}
*
\begin{bmatrix}
3 & 4 & 4 \\
1 & 0 & 2 \\
-1 & 0 & 3
\end{bmatrix}
=
\begin{bmatrix}
91 & 100 & \colorbox{grey}{83} \\
* & * & * \\
* & * & * \\
\end{bmatrix}
$$

For moving down to the next row, instead of jumping down by one pixel, we jump by the stride number, 2.

$$
\begin{bmatrix}
2 & 3 & 7 & 4 & 6 & 2 & 9 \\
6 & 6 & 9 & 8 & 7 & 4 & 3 \\
\colorbox{grey}{3^{3}} & \colorbox{grey}{4^{4}} & \colorbox{grey}{8^{4}} & 3 & 8 & 9 & 7 \\
\colorbox{grey}{7^{1}} & \colorbox{grey}{8^{0}} & \colorbox{grey}{3^{2}} & 6 & 6 & 3 & 4 \\
\colorbox{grey}{4^{-1}} & \colorbox{grey}{2^{0}} & \colorbox{grey}{1^{3}} & 8 & 3 & 4 & 6 \\
3 & 2 & 4 & 1 & 9 & 8 & 3 \\
0 & 1 & 3 & 9 & 2 & 1 & 4
\end{bmatrix}
*
\begin{bmatrix}
3 & 4 & 4 \\
1 & 0 & 2 \\
-1 & 0 & 3
\end{bmatrix}
=
\begin{bmatrix}
91 & 100 & 83 \\
\colorbox{grey}{69} & * & * \\
* & * & * \\
\end{bmatrix}
$$

This process is then repeated till all the needed values are obtained

$$
\begin{bmatrix}
2 & 3 & 7 & 4 & 6 & 2 & 9 \\
6 & 6 & 9 & 8 & 7 & 4 & 3 \\
3 & 4 & 8 & 3 & 8 & 9 & 7 \\
7 & 8 & 3 & 6 & 6 & 3 & 4 \\
4 & 2 & 1 & 8 & 3 & 4 & 6 \\
3 & 2 & 4 & 1 & 9 & 8 & 3 \\
0 & 1 & 3 & 9 & 2 & 1 & 4
\end{bmatrix}
*
\begin{bmatrix}
3 & 4 & 4 \\
1 & 0 & 2 \\
-1 & 0 & 3
\end{bmatrix}
=
\begin{bmatrix}
91 & 100 & 83 \\
69 & 91 & 127 \\
44 & 72 & 74 \\
\end{bmatrix}
$$

#### Input and Output Dimensions

The input and output dimensions here are governed by

$$\underset{\text{padding p}}{(n \times n)} *\underset{\text{stride s}}{(f \times f)} \rightarrow \left\lfloor \frac{n+2p-f}{s}+1 \right\rfloor  \times \left\lfloor\frac{n+2p-f}{s}+1\right\rfloor$$

Here, we only the boxed numbers being multiplied in the previous steps if it is fully contained within the image including the padding. For any parts outside the image, we do not perform any computation.

## Convolutions over several layers

In the real world most of the images we work with are in the form of 3D volumes, with two of the dimensions being the height and width of the image, and the third dimension being the channels of the image.

Consider a case where we want to detect features in an RGB image. In this example the dimensions of the image is $$6 \times 6 \times 3$$ and we want to convolve it with a $$3 \times 3 \times 3$$ filter with the third dimension corresponding to each of the color channels. Here the output we get is going to be of dimensions $$4 \times 4 \times 1$$.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/RGBConv1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Here we consider the filter to be a cube with 27 parameters in it. The mechanism here is similar to what we have seen before where we first place the convolution cube on the upper left position as shown, multiplying the corresponding number from the red, green, and blue channels with the filter's values, and adding them up to give us the first pixel of the output.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/RGBConv2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Similarly for the next step, we move the cube forward by one pixel and perform the same computations to obtain the next pixel value

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/RGBConv3.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

This process can be repeated for all the pixels of the image.

The advantages here are:
- If we want to check for edges in an image as per a given channel, we can focus on the given channel particularly while setting the other channels to zero.
- Edge detectors can be made to work irrespective of the color channel.
- This increases the number of choices we have in terms of the increased number of parameters, hence helping us detect more complex features.

We can also perform convolutions on a given image with multiple filters simultaneously to give us an output with the number of channels corresponding to the number of filters applied to the image. Using the same example as before, applying two filters to it would look something like the below

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/MultilayerRGBConv.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

This can be done with several filters simultaneously, giving a proportionally equal number of output channels corresponding to each feature being worked on.

## The Convolutional Neural Network Layer Structure

Now that we have seen how the convolution mechanism works, the last thing that is needed to create a convolutional neural network layer is to add a bias to our convolution values, alongside a non-linearity such as the ReLU or Sigmoid functions. An example can be seen below.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/ExampleLayer.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Comparing this to a single layer propagation in a standard neural network, the linear step equivalent is

$$Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$$

with the non-linear activation part given by

$$A^{[l]} = g(Z^{[l]})$$

The convolutional step equivalent of the same is as follows:
- The RGB image acts as the input $$A^{[0]}$$
- The filters act as the weights $$W^{[1]}$$
- The output of the convolution operation is the resultant $$Z^{[l]}$$ value which is fed to the activation function to obtain the output channels $$A^{[1]}$$.

Here, the number of parameters remain the same irrespective of the size of the input. These parameters instead depend on the dimensions and number of convolutional filters used.

## Notations for CNNs

- $$f^{[l]} =$$ filter size at layer $$l$$
- $$p^{[l]} =$$ amount of padding at layer $$l$$
- $$s^{[l]} =$$ stride at layer $$l$$
- $$n_{c}^{[l]} =$$ number of filters at layer $$l$$

- The input has the following properties
	- $$n$$ is the number of pixels
	- $$n_{c}$$ is the number of channels in the previous layer
	- $$n_{H}^{[l-1]} \times n_{W}^{[l-1]} \times n_{c}^{[l-1]}$$ is the dimension of the input image

- The output has the following dimensions

$$n_{H}^{[l]} \times n_{W}^{[l]} \times n_{c}^{[l]}$$

- the height of the output is given by

$$n_{H}^{[l]} = \left\lfloor \frac{n_{H}^{[l]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} + 1 \right\rfloor$$

- The same formula we apply for height also applies to the width of the output

- The size of each filter is $$f^{[l]} \times f^{[l]} \times n_{c}^{[l-1]}$$
- Activations have the dimensions $$n_{H}^{[l]} \times n_{W}^{[l]} \times n_{C}^{[l]}$$
- For mini batch GD, our output dimensions will be $$m \times n_{H}^{[l]} \times n_{W}^{[l]} \times n_{C}^{[l]}$$
- Weight dimensions: $$f^{[l]} \times f^{[l]} \times n_{c}^{[l-1]} \times n_{c}^{[l]}$$
- Bias dimensions: $$n_{c}^{[l]}$$ or $$1 \times 1 \times n_{c}^{[l]}$$

## A few general points about CNNs

Most of the work that goes into designing a ConvNet involves hyperparameter selection such as
- total size
- stride
- padding
- number of filters

In a CNN we usually start with larger images that gradually trend down in size as one goes deeper into the neural network. In turn, the number of channels increases.

The typical ConvNet consists of three layers:
- The Convolution Layer (Conv)
- Pooling Layer (Pool)
- Fully Connected Layer (FC)

Now that we have looked into the convolution layer, we will now look at the other layers at hand, and how they work towards building different CNN architectures.

## Pooling Layers

**Pooling layers** are used in CNNs to reduce the size of the representations in the network, helping speed up computation alongside making some detected features more robust.

### Max Pooling

Considering the example of a $$4 \times 4$$ input the output would be a $$2 \times 2$$ matrix.

$$
\begin{bmatrix}
\colorbox{darkblue}{1} & \colorbox{darkblue}{3} & \colorbox{blue}{2} & \colorbox{blue}{1} \\
\colorbox{darkblue}{2} & \colorbox{darkblue}{9} & \colorbox{blue}{1} & \colorbox{blue}{1} \\
\colorbox{green}{1} & \colorbox{green}{3} & \colorbox{red}{2} & \colorbox{red}{3} \\
\colorbox{green}{5} & \colorbox{green}{6} & \colorbox{red}{1} & \colorbox{red}{2}
\end{bmatrix}
\rightarrow
\begin{bmatrix}
\colorbox{darkblue}{9} & \colorbox{blue}{2} \\
\colorbox{green}{6} & \colorbox{red}{3}
\end{bmatrix}
$$

In this example of max pooling, the inputs are broken down into 4 distinct regions. The output here is obtained by taking the maximum of the corresponding shaded regions.

This is similar to taking a convolution filter of size 2 and stride 2 (which are the only parameters used in this filter), with the function applied taking only the maximum of the numbers within the filter position.

The idea of max pooling is that as long as a feature is detected in a given area, it will be preserved in the output, otherwise, the max of numbers in the region are relatively small.

Max pooling is the only type of CNN layer that has no parameters to learn and only two hyperparameters to learn.

#### Max Pooling Example

Consider a $$5 \times 5$$ image to which we want to apply a $$3 \times 3$$ filter, with filter size $$f=3$$ and step size $$s=1$$.

The output here will be a $$3 \times 3$$ matrix.

$$
\begin{bmatrix}
1 & 3 & 2 & 1 & 3 \\
2 & 9 & 1 & 1 & 5 \\
1 & 3 & 2 & 3 & 2 \\
8 & 3 & 5 & 1 & 0 \\
5 & 6 & 1 & 2 & 9 
\end{bmatrix}
\rightarrow
\begin{bmatrix}
* & * & * \\
* & * & * \\
* & * & *
\end{bmatrix}
$$

The formulas for finding the **output size** for the Convolution layers hold true in this case too. That is

$$
\left\lfloor \frac{n+2p-f}{s}+1 \right\rfloor
$$

The mechanism with which the filter is applied to the image is also the same as what is done in the case of convolution layers, shown below

$$
\begin{bmatrix}
\colorbox{grey}{1} & \colorbox{grey}{3} & \colorbox{grey}{2} & 1 & 3 \\
\colorbox{grey}{2} & \colorbox{grey}{9} & \colorbox{grey}{1} & 1 & 5 \\
\colorbox{grey}{1} & \colorbox{grey}{3} & \colorbox{grey}{2} & 3 & 2 \\
8 & 3 & 5 & 1 & 0 \\
5 & 6 & 1 & 2 & 9 
\end{bmatrix}
\rightarrow
\begin{bmatrix}
\colorbox{grey}{9} & * & * \\
* & * & * \\
* & * & *
\end{bmatrix}
$$

$$
\begin{bmatrix}
1 & \colorbox{grey}{3} & \colorbox{grey}{2} & \colorbox{grey}{1} & 3 \\
2 & \colorbox{grey}{9} & \colorbox{grey}{1} & \colorbox{grey}{1} & 5 \\
1 & \colorbox{grey}{3} & \colorbox{grey}{2} & \colorbox{grey}{3} & 2 \\
8 & 3 & 5 & 1 & 0 \\
5 & 6 & 1 & 2 & 9 
\end{bmatrix}
\rightarrow
\begin{bmatrix}
9 & \colorbox{grey}{9} & * \\
* & * & * \\
* & * & *
\end{bmatrix}
$$

$$
\begin{bmatrix}
1 & 3 & \colorbox{grey}{2} & \colorbox{grey}{1} & \colorbox{grey}{3} \\
2 & 9 & \colorbox{grey}{1} & \colorbox{grey}{1} & \colorbox{grey}{5} \\
1 & 3 & \colorbox{grey}{2} & \colorbox{grey}{3} & \colorbox{grey}{2} \\
8 & 3 & 5 & 1 & 0 \\
5 & 6 & 1 & 2 & 9 
\end{bmatrix}
\rightarrow
\begin{bmatrix}
9 & 9 & \colorbox{grey}{5} \\
* & * & * \\
* & * & *
\end{bmatrix}
$$

$$\vdots$$

$$
\begin{bmatrix}
1 & 3 & 2 & 1 & 3 \\
2 & 9 & 1 & 1 & 5 \\
1 & 3 & 2 & 3 & 2 \\
8 & 3 & 5 & 1 & 0 \\
5 & 6 & 1 & 2 & 9 
\end{bmatrix}
\rightarrow
\begin{bmatrix}
9 & 9 & 5 \\
9 & 9 & 5 \\
8 & 6 & 9
\end{bmatrix}
$$

In the case of 3D inputs, the outputs will have the same dimensions.

### Average Pooling

Average pooling, is similar in its mechanism to max pooling but the main difference here is that the average is taken instead of the maximum value

$$
\begin{bmatrix}
\colorbox{darkblue}{1} & \colorbox{darkblue}{3} & \colorbox{blue}{2} & \colorbox{blue}{1} \\
\colorbox{darkblue}{2} & \colorbox{darkblue}{9} & \colorbox{blue}{1} & \colorbox{blue}{1} \\
\colorbox{green}{1} & \colorbox{green}{3} & \colorbox{red}{2} & \colorbox{red}{3} \\
\colorbox{green}{5} & \colorbox{green}{6} & \colorbox{red}{1} & \colorbox{red}{2}
\end{bmatrix}
\rightarrow
\begin{bmatrix}
\colorbox{darkblue}{3.75} & \colorbox{blue}{1.25} \\
\colorbox{green}{4} & \colorbox{red}{2}
\end{bmatrix}
$$

Average pooling isn't as commonly used compared to max pooling.

An area where average pooling can be used is in very deep neural networks to collapse representations from large dimensions like $$7 \times 7 \times 1000$$ to $$1 \times 1 \times 1000$$.

### A Summary of Pooling

The hyperparameters for pooling are:
- $$f$$ : Filter size
- $$s$$ : Stride
- Type of pooling: Max or Average
- $$p$$ : padding (This is very rarely used)

The common filter size and stride values used are
- filter size $$f=2$$
- stride length $$s=2$$

Input and output of max pooling:

$$
n_{H} \times n_{W} \times n_{C} \rightarrow \left\lfloor \frac{n_{H} - f}{s} + 1 \right\rfloor \times \left\lfloor \frac{n_{W} - f}{s} + 1 \right\rfloor \times n_{C}
$$

Pooling has **no parameters to learn**. It is a **fixed function** that the neural network applies to the input layers.

##### A concern with respsct to pooling

Max-pooling layers can result in the loss of accurate spacial information.

## Why are convolutions used?

If we were to work with an image of dimensions $$32 \times 32 \times 3$$ implementing $$6$$ filters on it, this would give us a $$28 \times 28 \times 6$$ output.

In this case $$32*32*3=3072$$ and $$28*28*6=4704$$.

Using these values directly in a neural network with $$3072$$ inputs and a $$4704$$ unit output would give us a total of $$3072 \times 4076 \approx 14\text{ Million}$$, which is computationally expensive to train.

Considering that this is a relatively small image, one can only imagine how infeasible it would be to use this on images of larger dimensions.

Convolutional layers have a relatively smaller number of parameters. In the case of a $$5 \times 5$$ filter, including the bias parameter, we would be working with $$26$$ parameters. If we were to use six of these filters, then we would have $$(5*5*3 + 1)*6 = 456$$ parameters to work with.

The number of parameters in a CNN layer can be condensed into the given equation

$$\text{number of parameters} = (f^{[l]} \times f^{[l]} \times n_{C}^{[l-1]} + b)\times n_{C}^{[l]}$$

The reasons that ConvNets have an advantage over fully connected neural networks in terms of the number of parameters are:
- **Parameter Sharing**: A feature detector such as a vertical edge detector that is useful in one part of the input image can also be useful in other parts of the image.
    - In this case, suppose we have a $$3 \times 3$$ filter for vertical edge detection, the same can be used for detecting vertical edges in other parts of the image.
    - This holds true for lower level features of the network such as edges  as well as high level features such as eyes, nose, etc. for facial recongition.
    - As a result, the network will not need to learn different feature detectors for the same feature. 
- **Sparsity of connections**: Another advantage is that for each layer, the output values depend only on a small number of inputs.

It is mainly through these two mechanisms that a CNN is able to have fewer connections, hence allowing training on much smaller training sets while avoiding overfitting.

Another advantage of CNNs are that they are capable of capturing translation invariance. Here, if a given sample input is shifted in any direction, the filter will still be able to capture and identify the object in the image.

## Putting together a ConvNet

Suppose we wanted to implement a convolutional neural network with a labelled training set $$(x^{(1)},y^{(1)}), \dots , (x^{(m)},y^{(m)})$$.

A CNN structure will mainly comprise of convolution and pooling layers as the main body of the hidden layers.

These convolution and pooling layers are then followwed by fully connected layers that are formed by flattening the last convolution-like layer into a 1-dimension layer.

The final layer of the network, that is, the output layer will be a softmax output given by $$\hat{y}$$.

Each of the convolutional (conv) and fully connected (FC) layers will have various parameters in the form of weights $$W$$ and biases $$b$$. This is used to define the cost function.

$$J = \frac{1}{m} \sum\limits_{i=1}^{m}L(\hat{y^{(1)}},y^{(i)})$$

The training process in this case is similar to how we would perform it for an ANN, which is through using a gradient based root finding algorithm like gradient descent or any of its optimized versions to reduce the cost function $$J$$.

# Classical ConvNet Case Studies

There has been several decades of research done in the field of computer vision since the mid 1960s. The idea of convolutional operations was introduced in the 1980s via a paper by Toshiteru Homma, Les Atlas and Robert Marks II {% cite NIPS1987_98f13708 %}.

This led to further research into the field, finding rudimentary uses in handwriting detection and slowly branching out into other domains like medical image segmentation and breast cancer detection.

All this research snowballed into what we now recognize to be ConvNets, in the early 2000s, which forms the basis of modern computer vision.

Even though we are already aware of ConvNets and how they work it's always a good idea to explore different classical implementations of them to gain an idea of how they 

In this section we will look at some well known implementations of classical CNN architectures and how they work.

## LeNet-5

The **LeNet-5** architecture was introduced by Yann LeCun in a collaboratively written paper {% cite LeNet5 %} at AT&T Bell Labs back in 1998, and is one of the earliest neural networks developed.

This architecture was introduced as a better alternative to already existing pattern recognition systems that depended on hand-crafted feature extraction.

This paper introduces the concepts of gradient based learning for convolution filters to help them learn low level features, slowly moving up to more complex features when moving towards the output nodes.

The LeNet-5 architecture was trained on digitized $$32 \times 32$$ greyscale images of handwritten digits.

The main architecture of LeNet-5 is given below. Here, consider $$Cx$$ to be a convolutional layer, $$Sx$$ to be a pooling/subsampling layer and $$Fx$$ to be a fully connected layer. $$x$$ here refers to the layer index. While the original paper used average pooling for the sampling layers, the more modern variants of this architecture use max pooling.

| Layer          | Inputs         | No. of Filters | Kernel Size | Stride | Activation          | Outputs    |
| -------------- | -------------- | -------------- | ----------------- | ------ | ------------------- | ---------- |
| C1             | (32,32,1)      | 6              | 5                 | 1      | Sigmoid             | (28,28,6)  |
| S2 (Max pool)  | (28,28,6)      | 1              | 2                 | 2      | --                  | (14,14,6)  |
| C3             | (14,14,6)      | 16             | 5                 | 1      | Sigmoid             | (10,10,16) |
| S4 (Max pool)  | (10,10,16)     | 1              | 2                 | 2      | Flatten             | (5,5,16)   |
| F5             | (5,5,16) = 400 | 120            | 1                 | NA     | Sigmoid             | 120        |
| F6             | 120            | NA             | 84 units          | NA     | Sigmoid             | 84         |
| Output         | 84             | NA             | 10 units          | NA     | --                  | 1 of 10    |

<br>

In the second convolution layer C3, only 10 of the 16 convolution feature maps are connected to each of the inputs at a time. This is similar to dropout in a sense, playing a similar role of disrupting the network symmetry.

The nonlinearities used in the original paper were sigmoid and tanh. Modern implementations of this architecture make use of ReLU activations.

The original architecture did not implement a softmax layer and instead directly output values corresponding to each input.

By modern standards this network is relatively small with $$60,000$$ parameters. Modern neural networks work with 10 or 100 million parameters, with networks going upto a thousand times the size of this network.

The main pattern followed here is

$$\boxed{\text{Convolution Layer} \rightarrow \text{Pooling Layer}} \times 2 \rightarrow \boxed{\text{Fully Connected Layers}} \rightarrow \boxed{\text{Output}}$$

and, a trend here as one goes down the layers starting from the inputs, the height and width of each of the channels tend to decrease while the number of channels increases. There is no padding applied to any of the inputs.

## AlexNet

**AlexNet** {% cite AlexNet %} is one of the most influential papers in computer vision and was the first one to have an error less than 25% at the time. As of the writing of this blog, the paper for this network architecture

This network was designed by Alex Krizhevsky in collaboration with Ilya Sutskever and Geoffrey Hinton as a part of the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012.

This architecture here resembles LeNet-5 to a certain extent but has a significantly larger number of layers. 


The network architecture is as follows

| Layer            | Inputs           | No. of Filters | Kernel Size | Stride | Activation          | Outputs     |
| ---------------- | ---------------- | -------------- | ----------------- | ------ | ------------------- | ----------- |
| C1               | (227,227,3)      | 96             | 11                | 4      | ReLU                | (55,55,96)  |
| S2 (Max pool)    | (55,55,96)       | 1              | 3                 | 2      | --                  | (27,27,96)  |
| C3 (Same Conv.)  | (27,27,96)       | 256            | 5                 | 1      | ReLU                | (27,27,256) |
| S4 (Max pool)    | (27,27,256)      | 1              | 3                 | 2      | --                  | (13,13,256) |
| C5 (Same Conv.)  | (13,13,256)      | 384            | 3                 | 1      | ReLU                | (13,13,384) |
| C6 (Same Conv.)  | (13,13,256)      | 384            | 3                 | 1      | ReLU                | (13,13,384) |
| C7 (Same Conv.)  | (13,13,256)      | 256            | 3                 | 1      | ReLU                | (13,13,256) |
| S8 (Max pool)    | (13,13,256)      | 1              | 3                 | 2      | Flatten             | (6,6,256)   |
| F9               | (6,6,256) = 9216 | NA             | 4096 units        | NA     | ReLU, dropout = 0.5 | 4096        |
| F10              | 4096             | NA             | 4096 units        | NA     | ReLU, dropout = 0.5 | 4096        |
| Softmax (output) | 4096             | NA             | 1000 units        | NA     | --                  | 1 of 1000   |

<br>

This architecture utilized the ReLU activation function as opposed to the sigmoid and Tanh functions which aided in its increased performance.

A lot of the dimensions in the paper are different from what it is supposed to be as per Andrej Karpathy. This is so that the calculations come out right.

This network architecture was made to train using two GPUs with code written for them to communicate with each other.

The number of parameters here is $$60,000,000$$ as opposed to LeNet-5's $$60,000$$ parameters.

**Local response normalization** is implemented here to aid in generalization. This is done despite the fact that ReLU doesn't need input normalization to prevent saturation.

The structure pattern followed here is as follows

$$\boxed{\text{CNN} \rightarrow \text{Normalization} \rightarrow \text{Pooling}} \times 2 \rightarrow \boxed{\text{CNN}} \times 3 \rightarrow \boxed{\text{Pooling}} \rightarrow \boxed{\text{FC with dropout}} \rightarrow \boxed{\text{Output}}$$

## VGG-16

**VGG-16** was proposed by Karen Simonyan and Andrew Zisserman in 2014 for the ILSVRC. Its name stands for **Visual Geometry Group 16** was introduced by the Visual Geometry Group at the University of Oxford. The number in its name stands for the number of layers in the network, 16 layers of which 13 are convolutional layers and 3 are fully connected layers.

This model achieved a $$92.7\%$$ accuracy on the ImageNet dataset, containing 14 million images with $$1000$$ classes.

The inputs in this case, similar to the AlexNet model takes in $$(224,224,3)$$ images to give one of $$1000$$ possible outputs.

This network is well known for having a much simpler architecture with a focus on having convolution layers at a fixed size of $$f=3$$, with a stride of $$s=1$$ with same padding applied at every one of them. The max pooling layers were also fixed at the dimension of $$f=2$$ with a stride $$s=2$$.

The architecture here is as follows

| Layer            | Inputs            | No. of Filters | Kernel Size | Stride | Activation          | Outputs       |
| ---------------- | ----------------- | -------------- | ----------------- | ------ | ------------------- | ------------- |
| C1 (Same Conv.)  | (224,224,3)       | 64             | 3                 | 1      | ReLU                | (224,224,64)  |
| C2 (Same Conv.)  | (224,224,64)      | 64             | 3                 | 1      | ReLU                | (224,224,64)  |
| S3 (Max Pool)    | (224,224,64)      | 1              | 2                 | 2      | --                  | (112,112,64)  |
| C4 (Same Conv.)  | (112,112,64)      | 128            | 3                 | 1      | ReLU                | (112,112,128) |
| C5 (Same Conv.)  | (112,112,128)     | 128            | 3                 | 1      | ReLU                | (112,112,128) |
| S6 (Max Pool)    | (112,112,128)     | 1              | 2                 | 2      | --                  | (56,56,128)   |
| C7 (Same Conv.)  | (56,56,128)       | 256            | 3                 | 1      | ReLU                | (56,56,256)   |
| C8 (Same Conv.)  | (56,56,256)       | 256            | 3                 | 1      | ReLU                | (56,56,256)   |
| C9 (Same Conv.)  | (56,56,256)       | 256            | 3                 | 1      | ReLU                | (56,56,256)   |
| S10 (Max pool)   | (56,56,256)       | 1              | 2                 | 2      | --                  | (28,28,256)   |
| C11 (Same Conv.) | (28,28,256)       | 512            | 3                 | 1      | ReLU                | (28,28,512)   |
| C12 (Same Conv.) | (28,28,512)       | 512            | 3                 | 1      | ReLU                | (28,28,512)   |
| C13 (Same Conv.) | (28,28,512)       | 512            | 3                 | 1      | ReLU                | (28,28,512)   |
| S14 (Max pool)   | (28,28,512)       | 1              | 2                 | 2      | --                  | (14,14,512)   |
| C15 (Same Conv.) | (14,14,512)       | 512            | 3                 | 1      | ReLU                | (14,14,512)   |
| C16 (Same Conv.) | (14,14,512)       | 512            | 3                 | 1      | ReLU                | (14,14,512)   |
| C17 (Same Conv.) | (14,14,512)       | 512            | 3                 | 1      | ReLU                | (14,14,512)   |
| S18 (Max pool)   | (14,14,512)       | 1              | 2                 | 2      | Flatten             | (7,7,512)     |
| F19              | (7,7,512) = 25088 | NA             | 4096 units        | NA     | ReLU, dropout = 0.5 | 4096          |
| F20              | 4096              | NA             | 4096 units        | NA     | ReLU, dropout = 0.5 | 4096          |
| Softmax (output) | 4096              | NA             | 1000 units        | NA     | --                  | 1 of 1000     |

<br>

This network has a total of $$138,000,000$$ parameters.

# Residual Networks (ResNets)

When training a deep neural network the training is much slower as opposed to shallow networks.

Imagine we have to train a neural network on a large amount of data.{% cite BryceResNets %} When initializing the network, we know that the weights are randomly initialized, and as the inputs are fed into each layer, they are multiplied with these randomly generated weights and fed into activations. This is done several times such that towards the output layers, we get random noise as none of the information we receive is related to the input.

Now, when going through the backpropagation process, we compute the loss on the output and propagate it back through the network. We yet again face the same issue as before as the loss value later into the network do not inform the layers much about the inputs as their own inputs are noise. Hence, the gradients are also effectively scrambled and propagate noise back towards the input layers, even compounding it due to multiplications with weight matrices.

This leads to a lot of problems when training the network as the backprop algorithm struggles to learn from the inputs. In addition the problem here isn't even something like overfitting which would have been diagnosed easily. 

This is where **Residual Networks** or **ResNets** come into the picture.

ResNets {% cite BruntonResNets %} are one of the most important deep neural network architectures in the modern day. They find uses in several fields like image classification and ordinary differential equations. They are also one of the main ingredients used in transformers, which find use in large language models such as ChatGPT.

The main idea here is to create a way for data to arrive at later layers, hence making their inputs more meaningful, and also for loss gradients and their updates to have some meaning.

This is done by introducing the idea of **skip connections** which allow for data to be propagated deeper into the network.

## The Residual Block

ResNets are built upon a **residual block**, where a few layers of the network are grouped up into blocks, and the input activations of these blocks are both fed through each of its layers and also around it.

Consider a neural network modelling an input/output function with inputs $$x$$ and outputs $$y$$. The relationships can be shown as

$$y = f(x)$$

In ResNets, instead of trying to learn the direct relationship mapping, we create a copy of the input $$x$$ and only model the difference between them and $$y$$ given as

$$y = x + f(x)$$

This is because most of the features in the output $$y$$ is already present in the input and there is a only a small part in the form of the residual $$f(x)$$ capturing all the features in the input.

Here, the **residual** refers to the *minute changes* that occur to the inputs as one traverses down the network as opposed to the whole feature itself.

Consider the layers below taken from a several layer deep neural network.

<div class="row justify-content-center mt-3">
    <div class="col-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/two_layers.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The activations in these layers would look like

$$a^{[l]} \rightarrow \underbrace{Z^{[l+1]} = W^{[l+1]}a^{[l]} + b^{[l+1]}}_{\text{linear operator}} \rightarrow \underbrace{g(Z^{[l+1]}) = a^{[l+1]}}_{\text{Activation}} \rightarrow \underbrace{Z^{[l+2]} = W^{[l+2]}a^{[l+1]} + b^{[l+2]}}_{\text{linear operator}} \rightarrow \underbrace{g(Z^{[l+2]}) = a^{[l+2]}}_{\text{Activation}}$$

Here we see that the information flow goes all the way through from $$a^{[l]}$$ to $$a^{[l+2]}$$. We call this the **main path** for the given set of layers.

In the case of a residual network, the value for $$a^{[l]}$$ is copied much further into the neural network and added to $$Z^{[l+2]}$$, before the non-linearity. This is known as the **shortcut** as shown below.

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/calculation_shortcut.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Looking at the neural network architecture, this would look like

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/two_layer_res.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

A slightly more explicit look into this would be something like

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/skip_connection.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Now, the skip connections here give the data two paths to follow, one through the block and one skipping these intermediate layers and finding use deeper in the network.

This essentially helps train deeper neural networks without forgetting what the original inputs are towards the final stages of the network.

ResNets are built using several residual blocks, stacked together to form a deep network

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/plain_conn.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/res_conn.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A neural network made with plain connections and a residual network with skip connections.
</div>

The outputs of the block are combined with the input to the block in the form of a simple function that passes gradients along undisturbed by either
- **Adding** the tensor of inputs and tensor of outputs element-wise.
- **Concatenating** the tensors

Here, the latest activation acts as a residual added on to the previous information passed by the skip layer. We only focus on only modelling these residuals which capture the changes from the inputs and outputs of the residual layer. 

<!-- In theory, as the neural networks get deeper, they should perform better on the training set. However, in practice, deep networks cause the optimization algorithm to struggle to train, hence leading to a rise in training error. ResNets help mitigate this, allowing the training error to keep going down as the network is trained. -->

<!-- By taking activations from the inputs and earlier layers of the network and allowing them to influence layers deeper into the neural network, the vanishing and exploding gradient problems are mitigated, hence allowing deeper networks to be trained. We will now look at why this works. -->

## How ResNets Work

Consider a plain network as shown with $$X$$ being the inputs to a big neural network with $$a^{[l]}$$ as the output activations.

$$X \rightarrow \boxed{\text{Big NN}} \rightarrow a^{[l]}$$

Consider two more layers to be added to it , making it deeper.

$$X \rightarrow \boxed{\text{Big NN}} \xrightarrow{a^{[l]}} \text{layer }[l+1] \rightarrow \text{layer }[l+2] \rightarrow a^{[l+2]}$$

According to the ResNets {% cite DBLP:journals/corr/HeZRS15 %} paper, there exists a solution *by construction* to the deeper model

The added layers are *"identity mapping"* and the other layers are copying from the learned shallower model which is acting as the "identity" here.

Now, considering a skip connection from $$a^{[l]}$$ to $$layer[l+2]$$, we compare the equations for both cases 

<table>
<tr>
<th> Plain Neural Network </th>
<th> ResNet </th>
</tr>
<tr>
<td>
$$
\begin{align*}
Z^{[l]} &= W^{[l]}A^{[l-1]} + b^{[l]} \\
A^{[l]} &= g(Z^{[l]}) \\
Z^{[l+1]} &= W^{[l+1]}A^{[l]} + b^{[l+1]} \\
A^{[l+1]} &= g(Z^{[l+1]}) \\
Z^{[l+2]} &= W^{[l+2]}A`^{[l+1]} + b^{[l+1]} \\
A^{[l+2]} &= g(Z^{[l+2]}) \\
\end{align*}
$$
</td>
<td>
$$
\begin{align*}
Z^{[l]} &= W^{[l]}A^{[l-1]} + b^{[l]} \\
A^{[l]} &= g(Z^{[l]}) \text{ (Skip connection created)}\\
Z^{[l+1]} &= W^{[l+1]}A^{[l]} + b^{[l+1]} \\
A^{[l+1]} &= g(Z^{[l+1]}) \\
Z^{[l+2]} &= W^{[l+2]}A^{[l+1]} + b^{[l+1]} \text{ (Skip connection added)}\\
A^{[l+2]} &= g(Z^{[l+2]} + A^{[l]})
\end{align*}
$$
</td>
</tr>
</table>

<br>

If some amount of regularization or weight decay was applied here, the values of $$Z^{[l+2]}$$ would shrink down to zero or values close to zero. The effects of this would have are:

- For the plain model, the output activation $$A^{[l+2]}$$ would be set to a value biased towards zero as opposed to a value that would be more representative of the previous inputs. Hence, the nodes in the given layer would take a longer time to train and also have a lower accuracy.

- In the case of ResNets, we focus on building information on top of the "identity" formed by pre-existing layers of the network. So, instead of being shrunk to zero, the given layer's output would consist of the additional information, or "residuals" learned by the layers in between the skip layers and the output.

The idea is that if we have a shallower model learning a function, we should also be able to train a deeper model to learn the same function by copying the original five layers' values and using the additional layers to learn residuals that are added on top of the identity function.

The plain model does not learn the identity function because most of the weights are initialiized close to zero, hence causing any sort of regularization to bias the weights towards zero.

Hence, the idea of identity functions makes it easy for our residual blocks to train, making it so that adding the additional layers to the network will not have as much of an impact on its ability to learn.

#### Note

When building ResNets it is assumed that the dimensions of the skip connection and the output of the residual block have the same dimension.

Hence, we **must** use same convolutions in the residual network with $$1 \times 1$$ strides to ensure the same input and output dimensions.

In the occasion that the dimensions are different, we must add a matrix $$W_{s}$$ to transform the dimensions of the output to be the same as the input.

Here, $$W_{s}$$ can be a set of parameters to be learned or a matrix that implements zero padding.

If we also want to match up the number of channels, we use $$1 \times 1$$ convolutions. 

## Advantages of ResNets

##### 1. Skip blocks augment the existing data

As the skip connection passes the input down to the later layers of the neural network, the main network can focus on figuring out what information it has to add on top of the inputs as opposed to the whole input itself. This makes subsequent processing easier.

- If we are concatenating the inputs, it is passed through unchanged along with the outputs. If added, the value being sent is mostly unchanged as the weights are centered around zero.
- As a result of this each block learns a simpler task and has better access to information to learn from.

##### 2. Shorter gradient paths

Due to skip connections, the gradients have shorter paths to follow to get to each layer of the network. This is because each block has one path going through each of the layers and one path around them, and the gradients go back through both of them. Hence, any layer in the network will have a shorter path which the loss gradients can arrive to and usefully update the given layer's computation

-  As a result, in the initial epochs when the training results are not informative, we can still get useful updates from the shorter paths and obtain decreasing loss right away.
- In addition, over time, as the later blocks in the network start computing more useful functions, the information along the direct path becomes more informative, hence leading to a continued decrease in the gradients, outperforming the plain network.
- To sum it up, this leads to **faster training**.

##### 3. Modularity

An additional advantage that ResNets have is modularity. Since each of the blocks have similar structures, we can short circuit around blocks during training, making it easier to add more blocks and deepen the network.

- Most papers have variations on the same network with different numbers of blocks with different tradeoffs between the resources needed to train and the resulting model's predicting power.

## Concerns

##### 1. Shape Mismatch

If we want to combine the inputs and outputs of a ResNet, we need to ensure that the input and output shapes match up. For normal activation layers like dense networks, we might end up with different numbers of neurons per layer.

Hence, when adding the inputs to the outputs of the first block, we need an operation to reshape the inputs or only add the inputs to a part of the block's output.

With ConvNets, we will have to ensure that the height and width of the convolutional layers can be combined with the input height and width.

##### 2. Parameter Explosion

Another concern with ResNets is that with repeated concatenation at the output blocks, we can get a large activation tensor, hence leading to an explosion in the number of parameters. To counteract this, we prefer using addition to concatenation with matching input shapes.

# Shape Matching Convolutional blocks

A lot of implementations of ConvNets require for us to ensure that the output shapes of convolutional blocks are in a particular shape. One way to ensure this is to use $$1 \times 1$$ strides and same padding to ensure the width and height of the image are preserved.

However, if we want to match up the number of channels, we use $$1 \times 1$$ convolutions.

# $$1 \times 1$$ Convolutions (Networks in Networks)

$$1 \times 1$$ convolutions are useful when designing ConvNet architectures.

A $$1 \times 1$$ filter would be something like

$$
\begin{bmatrix}
1 & 2 & 3 & 6 & 5 & 8 \\
3 & 5 & 5 & 1 & 3 & 4 \\
2 & 1 & 3 & 4 & 9 & 3 \\
4 & 7 & 8 & 5 & 7 & 9 \\
1 & 5 & 3 & 7 & 4 & 8 \\
5 & 4 & 9 & 8 & 3 & 5 \\
\end{bmatrix}
*
\begin{bmatrix}
2
\end{bmatrix}
= 
\begin{bmatrix}
2 & 4 & 6 & \dots & &  \\
 \\
 \\
 \\
 \\
 \\
\end{bmatrix}
$$

Here we are effectively multiplying every pixel of the given image by $$2$$.

For a $$6 \times 6 \times 32$$ image, the filter goes through each of the $$36$$ pixels of the image, taking the element-wise product with the $$32$$ pixels in the same position along the channels, hence giving us a real number output to which we apply our non-linearities.

This is equivalent to a single neuron with $$32$$ nodes, multiplying each of them in place with $$32$$ weights, adding them up and applying our non-linearity to them.

For multiple filters. it is equivalent to having multiple units, taking all the numbers in a slice and building them up to an output.

This is called a $$1 \times 1$$ convolution or network in network.

### Alternate explanation

Using a $$1 \times 1$$ kernel on a multi-channel input gives an output block with the same height and width as the input. Each neuron in the output receives inputs from just one pixel of the input but also receives information from the entire depth of the particular channel it is pointed to.

Setting the number of filters helps us specify the depth of the output, and each neuron of the output can gets its inputs from the neurons across the depth of the previous layer. This is similar to adding a single dense layer to make the input and output shapes match up.

## Uses of $$1 \times 1$$ convolutions

Consider the given case below

$$28 \times 28 \times 192 \xrightarrow[CONV 1 \times 1 \;\; (32)]{ReLU} 28 \times 28 \times 32$$

To shrink the height and width of the volume above we would normally use a pooling layer, but we have no straightforward way to reduce the number of channels.

Hence, in this case, to shrink the number of filters down to $$32$$, we use $$32$$ $$1 \times 1$$ filters with each filter having a dimensions of $$1 \times 1 \times 192$$. This will effectively result in shrinking the number of layers down to $$28 \times 28 \times 32$$, hence helping save on computations in the networks.

The network in network effect is the addition of a non-linearity, allowing the learning of more complex functions by adding yet another layer.

# Inception Networks

Consider us building a neural network from scratch. We would pick our layers in the form of convolution filters of varying dimensions or pooling layers.

An **inception network** does the job of all of the discussed layers til now while making the network architecture more complicated and performing well.

Inception networks were developed by Google with the initial name of GoogleNet. Its name is inspired by the "We need to go deeper" meme from the movie Inception, and the meme is actually referenced in the paper, which is pretty funny.

Consider the case of a $$28 \times 28 \times 192$$ input. In an inception layer we do all of the operations at once.

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/inception_example.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An Inception layer
</div>

In the image above the inception network is performing the following operations
- A $$1 \times 1$$ convolution to give a $$28 \times 2 \times 64$$ output.
- A $$3 \times 3$$ same convolution to give a $$28 \times 28 \times 128$$ output.
- A $$5 \times 5$$ same convolution to give a $$28 \times 28 \times 32$$ output.
- A max pooling layer with padding to give us a $$28 \times 28 \times 32$$ output.

All of these are stacked one on top of the other to give us a composite output of dimensions $$28 \times 28\times 256$$.

The idea of an inception network is to be a swiss army knife to which we can put in our inputs to obtain the outputs exactly as per our needs. This network performs all the operations needed at once and concatenates them, allowing the network to learn the parameters to use alongside the filter sizes and pools needed.

The main drawback of inception layers is that the **computation cost** tends to be higher.

## Inception V1

The first iteration of the Inception network {% cite DBLP:journals/corr/SzegedyLJSRAEVR14 %} was created by Christian Szegedy as a part of the ILSVRC14 challenge.

It was mainly developed as an efficient neural network architecture compared to previous implementations, using upto 12 times fewer parameters than AlexNet.

As suggested by the meme it cites, the main idea here is to allow even deeper neural networks to be built with minimal effect on the network's performance.

The idea here is that given the rise of embedded and mobile computing, algorithm efficieny in terms of power and energy use are crucial alongside accuracy itself.

The inception network heavily makes use of the Network-in-Network approach to increase the representational power of the CNN. This serves two purposes:
- Dimension reduction to remove computational bottlenecks, which would limit network size.
- This also allows increasing the width of the network without significant impact on the performance. 

#### Motivations

The main way to increase the performance of a DNN is through increasing its size, both in depth and width considering the availability of data to work with. However this has some major drawbacks:
- Larger networks will need a larger number of parameters, hence leading to more chances for overfitting in case the number of training examples is limited.
- Increased network sizes also increase the amount of computation power needed, needing efficient distribution of computation resources. as opposed to directtly increasing the network size.

To tackle this, the solution followed is to implement a sparsely connected architecture (Including inside the convolutions) and to exploit the hardware to efficiently compute dense matrices.

#### Inception Architecture

The main idea followed here is finding a local sparse structure that can be approximated and covered by readily available dense components. This is built using convolutional building blocks.

Once the optimal local construction is found, it is repeated spatially. A layer by layer construction is followed where correlation statistics of the previous layer are analyzed and grouped into units with high correlation.

These groups/clusters are then used as units for the next layer and are connected to the units of the previous layer.

Using the same example as before, we could make a composition of the form
- A $$1 \times 1$$ convolution to give a $$28 \times 2 \times 64$$ output.
- A $$3 \times 3$$ same convolution to give a $$28 \times 28 \times 128$$ output.
- A $$5 \times 5$$ same convolution to give a $$28 \times 28 \times 32$$ output.
- A max pooling layer with padding to give us a $$28 \times 28 \times 32$$ output.

And then repeat this structure multiple times as per our need.

As the inception modules are stacked, the output correlation statistics will vary as features with higher abstraction are captured deep into the network. This leads to lesser spatial concentration and as a result, the computations become much more expensive. Max pooling, when added to the mix, leads to an increase in the number of outputs at each stage.

To combat this, **dimension reduction** is practiced at points where computational requirements go high. This is done with the help of $$1 \times 1$$ convolutions before the more expensive $$3 \times 3$$ and $$5 \times 5$$ convolutions. The resultant inception models are as follows.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/naive_inception.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Source, "Going Deeper With Convolutions" (Szegedy et al., 2014)
</div>

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dimension_reduced_inception.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Source, "Going Deeper With Convolutions" (Szegedy et al., 2014)
</div>

Hence, as a result we have a network architecture that alows us to build deep models without an increase in computational complexity and, in addition, allows visual information to be processed at various scales and aggregated so that the next layers can abstract features from them parallelly.

#### GoogLeNet

GoogLeNet made use of this Inception architecture to perform well under strict memory and computational budget, using only 5 million parameters, as opposed to AlexNet and VGGNet.

The resultant neural network architecture that was proposed by the team using these inception modules is shown below.

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/googleNet.jpg" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Source, "Going Deeper With Convolutions" (Szegedy et al., 2014)
</div>

As we can see here, there are two extra "auxillary networks" added to the architecture here aside from the inception layers. These serve the purpose of helping reduce the **vanishing gradient** problem that is introduced when training deep neural networks.

The auxillary classifiers make use of the outputs of the intermediate inception net layer outputs to generate softmax predictions of their own, based on which auxillary losses are obtained. These auxillary losses are used to augment the loss function during the training process. This is in the form

$$
J_{\text{total}} = J + (0.3 \times J_{\text{aux1}}) + (0.3 \times J_{\text{aux2}}) 
$$

These auxillary networks only serve the purpose of training the model and are removed during the inference process.

## Inception V2

The second version of the inception network {% cite DBLP:journals/corr/SzegedyVISW15 %} acknowledged the effectiveness of its older version, touting its effectiveness in solving big data problems with a lower effective cost. However, it also points out a major drawback of Inception v1 quoted directly from the paper:

> The complexity of the inception network makes it more difficult to make changes to the network.

The main aim of this paper was to upgrade this model such to increase its accuracy while also reducing the computational complexity.

Naively scaling up inception networks eliminates any computational gains that one would expect from such an architecture. In addition there was no explanation as to why certain design decisions were made with GoogLeNet, hence making it difficult to adapt to new use cases.

The main things tackled here are:
- **Avoiding representational bottlenecks:** The idea here is that instead of directly jumping from larger dimensions to smaller ones with large amounts of compression, the network must, instead, work towards gradually decreasing the representation size from the inputs to the outputs. Large jumps in the dimensions of the layers leads to a loss of information crucial for the network to train on, which is known as a **representational bottleneck**.
- **Using smart factorization methods** such as increasing the activations per tile, spatial aggregation and balancing the width and depth of the network to make the training process more efficient.

### Fixes proposed

##### 1. Factorizing convolutions with large filter sizes
The first change that was made to the network architecture was factorizing convolutions with larger filter sizes such as $$5 \times 5$$ or $$7 \times 7$$ to smaller sizes.

In this case the $$5 \times 5$$ convolutions are replaced with mini networks consisting of two $$3 \times 3$$ convolution layers as seen below.

<div class="row justify-content-center mt-3">
    <div class="col-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/original_inception.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5x5_factorized_inception.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The original inception architecture alongside the new change proposed (Szegedy et al., 2015). Note the filter marked by a red boundary.
    This is "Figure 5"
</div>


##### 2. Futher factorization of $$n \times n$$ filters to a combination of $$1 \times n$$ and $$n \times 1$$ convolutions

The inception module can further be changed by using asymmetric convolutions on the images, which would make each of them behave like a single layer fully connected with one another.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/1xn_nx1_variant.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The proposed (1,n) and (n,1) architecture (Szegedy et al., 2015). This is "Figure 6"
</div>

Taking the case of a $$3 \times 3$$ filter, each can be replaced with a $$3 \times 1$$ convolution followed by a $$1 \times 3$$ convolution.

This solution is $$33\%$$ cheaper than the full $$3 \times 3$$ convolution. For $$n \times n$$ filters, as $$n$$ increases in value, this can lead to drastic improvements in computational cost saving.

This idea is, however, not effective at the earlier layers of the network, but works well on medium grid sizes.

##### 3. Using higher dimension representations to increase the activations per tile

It was found that using higher dimension representations would allow for disentanglement of the features being worked on, hence leading to faster training. As a result another inception layer type was introduced with expanded filter bank outputs.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/expanded_filter_banks.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The proposed expanded filter banks architecture (Szegedy et al., 2015). This is "Figure 7"
</div>

This helped with removing representational bottlenecks.

### Network Architecture

Using the architectures proposed above, the final proposed neural network architecture is given below 

| type                   | patch size/stride  | input size                   |
| ---------------------- | ------------------ | ---------------------------- |
| conv                   | $$3 \times 3/2$$   | $$299\times 299 \times 3$$   |
| conv                   | $$3 \times 3/1$$   | $$149 \times 149 \times 32$$ |
| conv padded            | $$3 \times 3/1$$   | $$147 \times 147 \times 32$$ |
| pool                   | $$3 \times 3/2$$   | $$147 \times 147 \times 64$$ |
| conv                   | $$3 \times 3/1$$   | $$73 \times 73 \times 64$$   |
| conv                   | $$3 \times 3/2$$   | $$71 \times 71 \times 80$$   |
| conv                   | $$3 \times 3/1$$   | $$35 \times 35 \times 192$$  |
| $$3 \times$$ Inception | As in figure $$5$$ | $$35 \times 35 \times  288$$ |
| $$5 \times$$ Inception | As in figure $$6$$ | $$17\times 17 \times 768$$   |
| $$2 \times$$ Inception | As in figure $$7$$ | $$8\times 8 \times1280$$     |
| pool                   | $$8 \times 8$$     | $$8 \times 8 \times 2048$$   |
| linear                 | logits             | $$1 \times 1 \times 2048$$   |
| softmax                | classifier         | $$1 \times 1 \times 1000$$   |

<br>

## Inception V3

This version of Inception was proposed in the same paper as Inception V2 and has more or less the same upgrades that were implemented in Inception V2. The main improvements made here were:

- RMSProp Optimizer was implemented.
- $$7 \times 7$$ convolutions were factorized into three $$3 \times 3$$ convolutions.
- **Batch Normalization in the auxillary classifiers:** If we look back at the first version of inception networks, there were two auxillary classifiers introduced that were believed to help augment the stability, improve the convergence and mitigate the vanishing gradient problem.
    - However, it was later found that these did not necessarily help much in improving the convergence earlier in the training process. Instead they acted more as regularization for their respective layers.
    - Batch Normalizing these auxillary classifiers was found to lead to a better performance of the main network.
- Label smoothing was implemented as a regularizing component in the loss formula to prevent overfitting.

## Other implementations

There have been more implementations of the inception network such as Inception V4 and Inception-ResNet, which were introduced in the same paper {% cite DBLP:journals/corr/SzegedyIV16 %} in 2016. These dealt with the addition of residual connections to inception blocks to prevent the loss of computed values deeper into the networks.

As of now we will not look into these architectures as there is much more to talk about in computer vision, and focusing on these will take up a lot of time on the blog (which is long enough as is). It is always advised to read the related paper as it is well written and does a really good job at explaining what is going on.

# MobileNet

Deep neural networks which distinguish between a wide variety of labels generate a large amount of features, especially as we go deeper into the layers. This becomes computationally expensive because the number of channels of the filters also increase as one uses more filters while going` deeper into the network.

MobileNet {% cite DBLP:journals/corr/HowardZCKWWAA17 %} is another foundational neural network architecture designed with a focus on solving the computational scalability issues mentioned before while being lightweight and efficient for computer vision based applications in low compute environments like mobile phones and microprocessors.

The idea here is that each filter need not necessarily look at every single channel of the input at once. We can instead make each filter work on the input channels one by one and combine them into a single output.

The main motivations for this architecture are:
- Low computational costs
- Usage of models with this architecture in mobile phones and embedded systems.
- Introducing the idea of depth-wise separable convolutions.

The idea of this class of network architecture is to allow a deep learning model developers to choose a small network according to the constraints of the machine it is being run on. This is meant to optimize for latency (speed) while yielding small networks.

This architecture implements the idea of "depthwise separable convolutions" which were previously used in inception networks but were never given a proper name. These help reduce the computation costs at the lower layers of the network.

## What MobileNet aims to solve

Looking back at normal convolutions as seen from the example below

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/RGBConv1.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Assuming input image dimensions to be $$n \times n \times n_{c}$$ being put through a convolution filter of dimension $$f \times f \times n_{c}$$. Each side of the output obtained here, taking $$p$$ as the padding and $$s$$ as the stride, would be:

$$\frac{n + 2p -f}{s}$$

If we were to compute the cost here, it would be proportional to the form

$$\text{Computational cost} = \text{# filter params } \times \text{ # filter positions } \times \text{ # of filters}$$

with the computational costs depending multiplicatively on the number of input channels, number of output channels, kernel size and feature map/image size.

So, taking the number of filters to be $$5$$ with $$0$$ padding and stride $$1$$, the computational cost would be

$$\text{Computational Cost} = (3 \times 3 \times 3) \times (4 \times 4) \times 5 = 2160$$

Though this may seem small, when building deeper networks, the computational costs of these calculations add up, hence impacting the performance of the network.

The **depthwise separable convolutions** were introduced to help with this issue.

## Depthwise Separable Convolutions

The depthwise separable convolution consists of two stages
- The **depthwise** convolution stage, followed by
- The **pointwise** convolution stage

and the order of computing these are shown below

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/depthwise_separable_convolution.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The depthwise part applies a single filter to each input channel and the pointwise convolution applies a $$1 \times 1$$ convolution to combine the outputs of the depthwise convolution.

The idea here is to split up the interaction between the number of output channels and kernel size.

We will now look at what occurs during each stage.

#### Depthwise Convolution

Here, for a $$n \times n \times n_c$$ input, where $$n_c$$ is the number of channels, we make use of a filter of dimensions $$f \times f$$ applied to each channel of the input.

Here, one of each of the filters is applied to each of the corresponding input channels such that, the red channel is convolved with one filter, the green with another one and so on, till we get a $$n_{out} \times n_{out} \times n_c$$ output.

The computation cost here is proportional to

$$(3 \times 3) \times (4 \times 4) \times 3 = 432$$

As we can see here, this depthwise convolution action is much more efficient than a standard convolution, however it is only filtering the input channels in certain manners. It isn't combining these to create a feature map. This is where **pointwise convolutions** come into the picture.

#### Pointwise Convolution

Naively concatenating the outputs of the depthwise convolution layer leads to an issue where each layer of the output would only depend on one of the convolution action of the input. This pattern will repeat deeper into the network, hence not giving us as effective of a result as a normal convolution layer.

To combat this we make use of $$1 \times 1$$ convolutions known as **pointwise convolutions** corresponding to the number of filter we need to act on the input.

The pointwise convolution takes the $$n_{out} \times n_{out} \times n_{c}$$ convolution obtained from the depthwise convolution part and applies number of $$1 \times 1$$ convolutions equal to the number of filters being applied to the input.

Taking the same example as before where we have a $$6 \times 6 \times 3$$ input with a filter size of $$3$$, stride of $$1$$, padding $$0$$ and number of filters as $$5$$, the computational cost here would be proportional to

$$(1 \times 1 \times 3) \times (4 \times 4) \times 5 = 240$$

#### Comparing the cost with normal convolutions

Adding up the depthwise and pointwise convolution parts, we get the total computational costs to be proportional to 

$$\underbrace{((3 \times 3) \times (4 \times 4) \times 3)}_{\text{depthwise}} + \underbrace{((1 \times 1 \times 3) \times (4 \times 4) \times 5)}_{\text{pointwise}} = 672$$

which is much lower compared to that of a normal convolution which is

$$(3 \times 3 \times 3) \times (4 \times 4) \times 5 = 2160$$

Here we see that the depthwise separable convolutions are $$31 \%$$ as computationally expensive as a normal convolution layer.

## Generalized formulae for computational costs and their ratios

Generalizing the formula for computation costs, we take the following:
- $$n_c$$ as **number of channels**
- $$f$$ as **filter dimension**
- $$n_f$$ as **number of filters** being used
- $$n_{out}$$ as the **output dimensions**, which can also be seen as the **number of filter positions** on the input image.

and get

$$
\begin{align*}
& \boxed{\text{Computations}_{\text{MobileNet}} = (f \times f \times n_c \times n_{out} \times n_{out}) + (n_c \times n_{out} \times n_{out} \times n_f)} \\
& \boxed{\text{Computations}_{\text{ConvNet}} = f \times f \times n_c \times n_{out} \times n_{out} \times n_f} \\
& \boxed{\text{Computations Ratio} = \frac{\text{Computations}_{\text{MobileNet}}}{\text{Computations}_{\text{ConvNet}}} = \frac{1}{n_f} + \frac{1}{f^2}}
\end{align*}
$$

## MobileNet Architecture

The MobileNet architecture is fairly straightforward. Wherever we would have particularly heavy computations involving convolution operations, we use depthwise separable convolutions.

The MobileNet V1 paper implemented this block 13 times from the raw image inputs to the final classification prediction vis a pooling layer, fully connected layer and a softmax output.

## Experiments done on the MobileNet architecture

### Model Shrinking Hyperparameters: The Width Multiplier

A concept introduced with this hyperparameter to make the network much less computationally expensive was the **width multiplier**, $$\alpha$$.

The width multiplier's job is to ensure that at each layer, the network remains thin. It's value is set such that $$\alpha \in (0,1]$$.

So, considering the number of input channels to be $$n_c$$, the computations become proportional to

$$
\text{Computations}_{\text{MobileNet}} = (f \times f \times \alpha n_c \times n_{out} \times n_{out}) + (\alpha n_c \times n_{out} \times n_{out} \times \alpha n_f)
$$

The computational costs here are reduced by $$\alpha^2$$ times.

The $$\alpha$$ values lead to a computation and size tradeoff with a smooth accuracy dropoff until $$\alpha = 0.25$$ after which the model becomes too small.


### Model Shrinking Hyperparameters: The Resolution Multiplier

The **resolution multiplier** is another hyperparameter here which helps reduce the computational costs of the neural network. It is denoted by $$\rho$$ such that $$\rho \in [0,1)$$.

The resolution multiplier is applied to the image inputs to effectively decrease their resolution. The effective computation considering this hyperparameter along with the width multiplier is:

$$
\text{Computations}_{\text{MobileNet}} = (f \times f \times \alpha n_c \times \rho n_{out} \times \rho n_{out}) + (\alpha n_c \times \rho n_{out} \times \rho n_{out} \times \alpha n_f)
$$

The effective reduction in computational cost owing to this paramparameter is proportional to $$\rho^{2}$$. 


## MobileNet V2

The second version of the MobileNet architecture, MobileNet V2 {% cite DBLP:journals/corr/abs-1801-04381 %} was introduced by a group of researchers in 2017, which made two changes to its structure:
- It added a resudual connection from each of the units' input layer, passing the information directly to the later layers.
- It included an expansion layer before the depthwise separable convolution pair, known as a projection.

Ever since depthwise seprarble convolutions became a thing, many neural network architectures have adopted this idea as a drop-in replacement for standard convolutions. This led to a reduction in computational costs proportional to $$f^{2}$$ where $$f$$ is the filter dimension.

The original MobileNet depthwise separable convolution block implements a $$1 \times 1$$ filter followed by a depthwise convoltion of size $$n \times n$$ (in this case $$n = 3$$) and a pointwise convolution of dimensions $$1 \times 1$$. This effectively follows a $$\text{Wide} \rightarrow \text{Narrow} \rightarrow \text{Wide}$$ pattern when we look at the number of channels we are dealing with. As a result the 

MobileNet V2 sets the value of $$f$$ to $$3$$, hence leading to the computational costs being approximately $$9$$ times smaller than that of normal $$3 \times 3$$ convolution blocks with a minimal reduction in the accuracy.

### Model Architecture

The name of the block used here is the **bottleneck block**, and it is the basis of our model architecture. The main components of this block are a depthwise separable convolution with a residual connection and an additional Expansion layer before the depthwise separable layer. The structure of this block is shown below

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/MobileNetV2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The last step brings down the dimensions of inputs $$k$$ to $$k'$$ due to which it gets the name **projection** layer.

### What the Bottleneck Block Accomplishes

The main things that the bottleneck block aims to accomplish are:
- The pointwise convolution/projection reducces the dimensionality of the input features to a smaller set of values before extracting new features from them. Hence, this leads to an increase in efficiency of the computations while maintaining a good level of accuracy.
- Residual connections here help improve the flow of information in the network.
- The effective number of matrix multiplication actions being performed is reduced significantly

<!-- #### What are manifolds?

Before moving further we must understand what a manifold is.

When we have data at higher dimensions like 4, 5 and above, we cannot fathom what it would be like to visualize these as opposed to 2D and 3D spaces. To help with this, we need to reduce the dimensions of the dataset.

One way to do this is taking a projection of the data to allow visualizing it, such as in the case of Principal Component Analysis and other such methods.

Manifolds generalize linear frameworks like PCA to accommodate for non-linear data -->