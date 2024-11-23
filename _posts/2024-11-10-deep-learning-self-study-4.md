---
layout: post
title: Deep Learning - Object Detection
date: 2024-11-10 12:00:00
description: "Topics covered: Object Detection Algorithms."
tags: machine-learning deep-learning computer-vision
categories: deep-learning
toc:
  sidebar: left
bibliography: 2024-06-30-dl-1.bib
related_publications: true
pretty_table: true
---

Now that we have looked into the basics of ConvNets and how they tie into image recognition, we now have a look at one of the most interesting and widely researched fields in computer vision, **object detection**. This blog, similar to the ones preceeding it, is loosely based on the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by Andrew Ng, and will dive a bit deeper into the concepts discussed in his course content through direct paper reviews and practical experiments in Python.

<hr>

# Object Localization

**Object localization** is one of the main methods that differenciate object detection from image recognition.

In **image recognition** the focus is more on identifying what is present in a given image, for example, "Is a given image a cat or a dog?"

**Classification with localization**, however, is different in that not only do we have to put a label to identify whether a given image has a cat in it or not, but we should also be able to put a bounding box in the given picture to show where exactly the cat is located.

**Detection** on the other hand deals with finding several instances of a given object in the image and localizing them. This has a wide variety of use cases, particularly in autonomous vehicles where we would need to not only detect other cars but also motorcycles, people, pets and so on.

Classification and localization usually deal with a single object in the image we are trying to recognize and localize. The detection problem deals with several objects of different types being present in a given image. All three of the concepts mentioned here tie into and build upon each other.

Considering the example of classification, we would use a ConvNet to generate a set of vector features that when connected to a softmax output, outputs the prediction.

If we wanted to localize an object in the image, we change the network by adding a few more output units such that it generates a bounding boxes. The bounding boxes are parameterized with four numbers $$(bx,by,bh,bw)$$

Taking the upper left corner of the image as $$(0,0)$$ and the botton right as $$(1,1)$$, the specification of the bounding box needs
- The midpoint of the position where the object is present $$(bx,by)$$.
- The height $$bh$$ and width $$bw$$ of the bounding box

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cat_bounding_box.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An example of how a bounding box would be generated on an image.
</div>

Hence, the training set for this use case will not only contain the **label** for the image data, but also **the four numbers** signifying the box that bounds the location of the object in the image.

As a result. our supervised learning based network can tell us both, the label of the object in the image, and its location and relative size in the image.

## Defining the target

The target label of our dataset can be seen as the following vector

$$
y = \begin{bmatrix}
P_c \\ bx \\ by \\ bh \\ bw \\ c_1 \\ c_2 \\ \vdots
\end{bmatrix}
$$

The components here are as follows
- $$P_c$$: This is a probability value that informs the model whether an object from the given set of labels is present ($$1$$) or not ($$0$$). The not case here would be the "background".
- $$bx,by$$: the coordinates of the center of the object in the image relative to the upper left corner
- $$bh,bw$$: The height and width of the bounding box surrounding the object.
- $$c_1,c_2, \dots$$: These tell us the probability of the given object belonging to each class. In the case of classification with localization, at most one of these classes will appear in the picture.

So, given the image example above with the cat, if we we had a bunch of classes out of which $$c_1$$ represented a cat, then we'd get

$$
y = \begin{bmatrix}
1 \\ bx \\ by \\ bh \\ bw \\ 1 \\ 0 \\ \vdots
\end{bmatrix}
$$

For no object detected in the image, we would get

$$
y = \begin{bmatrix}
0 \\ ? \\ ? \\ ? \\ ? \\ ? \\ ? \\ \vdots
\end{bmatrix}
$$

Here the $$?$$ represents "don't care", since the the rest of the values don't matter to us if the given object is not present in the image 

## The Loss Function

Given that $$y$$ is the ground truth and $$\hat{y}$$ is the prediction by our model. For squared error, taking $$y_1$$ to be $$P_c$$ our loss will be

$$
L(\hat{y},y) = \begin{cases}
\begin{align*}
&(\hat{y}_1 - y_1)^2 + (\hat{y}_2 - y_2)^2 + \dots & \text{if }y_1=1 \\
&(\hat{y}_1 - y_1)^2 & \text{if }y_1=0
\end{align*}
\end{cases}
$$


In real implementations we can use different error types for different components, like MSE for the bounding box information, logistic regression loss for $$P_c$$, and a log-like feature loss for the class-related features (as they more or less behave like softmax loss).

# Landmark Detection

Given that we are able to output a bounding box for an object in our image based on a central point, this concept can be generalized such that we can just get the output of the $$x$$ and $$y$$ coordinates of several points on the image from the neural network that we deem important. These points are known as **landmarks**.

A use case for this can be in facial recognitiion, where we would want to locate parts of the face like the pupils of the eye, nose, etc. Suppose we wanted to find where the center of the lips are, we can instruct the neural network's final layer to give us two more outputs $$l_x$$ and $$l_y$$ corresponding to the position of the lips.

If we instead wanted, say, the edges of the mouth, we could instead tell the model to give us $$l_{1x}$$, $$l_{1y}$$, $$l_{2x}$$, $$l_{2y}$$ with each corresponding to the $$x,y$$ coordinates of the edges of the mouth.

Expanding upon this idea, we could introduce more and more points into the equation, hence allowing us to predict more nuanced information from the image such as, say, looking at several points on a person's mouth to see if they were smiling or frowning, determining which direction a person is looking at by looking at their eyes or the face shape, seeing if their eyes are opened or closed, and so on.

This can all be done with the help of multiple landmarks. We will need to generate a training set consisting of these landmarks and hence have a neural network tell us the key positions on the image.

This more or less forms the basis of how we recognize different aspects of images such as whether a person is happy or sad, maybe even trying to apply a filter on a person's face, and so on.

An obvious **drawback** we can see here is the need to go through the tedious process of labelling our data with all the relevant points.

# Object Detection

Now that we have seen how object localization and landmark detection works, we can use these concepts to implement **object detection**.

We can use ConvNets to perform object detection tasks using a mechanism known as the Sliding Windows Detection Algorithm.

Our dataset can consist of closely cropped examples of images belonging to different classes mixed in with images having no labelled objects in them. Using this, a ConvNet can train on these images to output a prediction $$\hat{y}$$.

## Sliding Window Object Detection

With a trained ConvNet, we can implement a sliding window mechanism on our test image to detect instances of a predicted object. The process of this algorithm is as follows

- Select a window size and input a small region of the image bounded by the given window into the ConvNet.
- Once you receive a prediction for the given region bounded by the rectangle, move onto another region next to the previous position, feed it into the ConvNet and perform a prediction again.
- Repeat this process for several regions on the image till every position has been covered. This can be seen to be working in a similar way to how convolution kernels move along an image with variable strides.
- Now, repeat this process with larger windows.

The idea here is that if there is an object that we are looking for present in the image, there will be a window where the window can output a $$1$$ for that region.

### Disadvantage of the Sliding Window approach

The main disadvantage here is the computational cost, as we are performing a prediction at several positions on our image with varying window sizes.

Using a coarse window stride can't reduce the effective number of windows we need to pass through the network, but this will lead to a tradeoff with the model performance. Similarly, using a small stride would lead to a huge number of regions being fed into the ConvNet, which will increase the computation time.

In classical approaches, these sliding windows were seen as a better approach since the models used were much more straightforward and less computationally taxing.

With ConvNets, this changes, and sliding windows prove to be slow and using larger strides would just lead to a lower accuracy. Hence, this implementation was changed for ConvNets.

# Sliding Windows with ConvNets

We will now see how we can implement sliding windows with Convolutional Networks.

$$2 \times 2 \times 400$$

$$24 \times 24$$