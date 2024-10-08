---
layout: post
title: Deep Learning - Basics of Computer Vision
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

This blog will cover one such domain that has been of great interest to me for the longest time: **Computer Vision**. A lot of the content of this blog is based off papers written by the original authors with the main structure of the blog being highly inspired by the [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning) portion of the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by Andrew Ng. The images in this blog were either made by hand, in [draw.io](https://app.diagrams.net/) or were generated through python. Without wasting any more time, let's begin.

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

| Layer          | Inputs         | No. of Filters | Kernel/layer Size | Stride | Activation          | Outputs    |
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

| Layer            | Inputs           | No. of Filters | Kernel/layer Size | Stride | Activation          | Outputs     |
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

| Layer            | Inputs            | No. of Filters | Kernel/layer Size | Stride | Activation          | Outputs       |
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

When training deep neural networks, the problem of vanishing and exploding gradients is one of the biggest issues that one faces. I'd highly suggest looking back at the [previous blog]({{site.baseurl}}{% link _posts/2024-07-22-deep-learning-self-study-2.md %}) to recap on what these problems pose.

However, to save you the time of going through that blog, the basic idea is that as we train a neural network, going through several layers of activations, the output values can either grow or shrink exponentially due to the weight matrices being scaled slightly smaller or larger with each layer that is passed, leading to a lot of performance issues when training the network.

Weight initialization is one method of mitigating this issue as had been discussed before, but this can only help so much. As the depth and complexity of neural networks go into the several hundreds, it can get difficult to avoid these issues even with proper weight initialization strategies.

This is where **Residual Networks** or **ResNets** come into the picture.

ResNets {% cite BruntonResNets %} are one of the most important deep neural network architectures in the modern day. They find uses in several fields like image classification and ordinary differential equations

They introduce the idea of implementing a **skip connection**. This is done by taking the activations of a layer and feeding them to a layer later down the line alongside the already existing activations from the previous layer. This is what helps us build deeper neural networks.

## The Residual Block

ResNets are built upon a **residual block** which we will be discussing now.

Consider a neural network modelling an input/output function with inputs $$x$$ and outputs $$y$$. The relationships can be shown as

$$y = f(x)$$

In resnets, instead of trying to learn the direct relationship mapping, we create a copy of the input $$x$$ and only model the difference between them and $$y$$ given as

$$y = x + f(x)$$

This is because most of the features in the output $$y$$ is already present in the input and there is a small residual $$f(x)$$ capturing all the little features in the input

Usually as we traverse down very deep neural network, it becomes hard to remember the original input $$x$$ over many layers. In many cases the output is similar to $$x$$ so it is important that it is not forgotten. ResNets help us train much deeper networks without forgetting the inputs at the final stages.

Here the residual refers to the minute changes that occur to the inputs as one traverses down the network as opposed to the whole feature itself.

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

This effectively allows the data to follow two different paths, allowing the information to skip several layers and find use in deeper parts of the neural network.

This essentially helps train deeper neural networks without without forgetting what the original inputs are towards the final stages of the network.

Here, the latest activation acts as a residual added on to the previous information, and we focus on only modelling these residuals which capture the changes from the inputs and outputs of the residual layer. 

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

When adding residual connections, they can either be **added** or **concatenated** with the linear part such that it forms a single tensor.

In theory, as the neural networks get deeper, they should perform better on the training set. However, in practice, deep networks cause the optimization algorithm to struggle to train, hence leading to a rise in training error. ResNets help mitigate this, allowing the training error to keep going down as the network is trained.

By taking activations from the inputs and earlier layers of the network and allowing them to influence layers deeper into the neural network, the vanishing and exploding gradient problems are mitigated, hence allowing deeper networks to be trained. We will now look at why this works.

## How ResNets Work

Consider a plain network as shown with $$X$$ being the inputs to a big neural network with $$a^{[l]}$$ as the output activations.

$$X \rightarrow \boxed{\text{Big NN}} \rightarrow a^{[l]}$$

Consider two more layers to be added to it along with a skip connection to make it deeper.

$$X \rightarrow \boxed{\text{Big NN}} \xrightarrow{a^{[l]}} \text{layer }[l+1] \rightarrow \text{layer }[l+2] \rightarrow a^{[l+2]}$$

The final ReLU activation which has outputs such that $$a \geq 0$$, is now changed such that

$$
\begin{align*}
a^{[l+2]} &= g(Z^{[l+2]} + a^{[l]}) \\
&= g(W^{[l+2]}a^{[l+1]} + b^{[l+2]} + a^{[l]})
\end{align*}
$$

Now, if we use L2 regularization for weight decay in this function, $$W^{[l+2]}$$ is shrunk. Hence, if the weight and bias at $$[l+2]$$ are zero, we get

$$a^{[l+2]} = g(a^{[l]}) = a^{[l]}$$

when using ReLU activations.

In this case, we see that the identity function is easy for our residual blocks to train on, hence making it so that adding the additional layers will not have as much of an impact on the neural network's ability to learn.