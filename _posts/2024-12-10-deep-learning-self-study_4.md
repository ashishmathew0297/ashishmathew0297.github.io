---
layout: post
title: Deep Learning - Object Detection
date: 2024-12-10 12:00:00
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

## Converting Fully Connected Layers into Convolutional Layers

Take the following network structure into consideration

| Inputs         | No. and Type of Filters | Kernel Size | Output     |
| -------------- | ----------------------- | ----------- | ---------- |
| (14,14,3)      | 16 (CONV)               | 5           | (10,10,16) |
| (10,10,16)     | 1  (MAX POOL)           | 2           | (5,5,16)   |
| (5,5,16)       | 400 UNITS  (FC)         | NA          | 400        |
| 400            | 400 UNITS  (FC)         | NA          | 400        |
| 400            | 4 UNITS  (SOFTMAX)      | NA          | 4          |


Now, we change the final softmax layer to show four numbers corresponding to the probabilities of the softmax output.

Taking the layer between the $$(5,5,16)$$ input from the MAX POOL layer to the $$400$$ unit fully connected output as an example:

- To convert the fully connected layer into a convolutional layer, we use $$400$$ $$5 \times 5$$ filters.
- This because when we implement the $$5 \times 5$$ filter, it is implemented as $$5 \times 5 \times 16$$ as it is looking at all the channels and must match the outputs.
- Each convolution operation imposes a $$5 \times 5$$ filter on top of the $$5 \times 5 \times 16$$ feature mapping, giving us a $$1 \times 1$$ output.
- We can say that the $$400$$ node fully connected layer is equivalent to a $$1 \times 1 \times 400$$ volume. This is mathematically the same as a fully connected layer as each node corresponds to a filter of $$5 \times 5 \times 16$$.

Similarly, for the second fully connected layer, we perform a $$1 \times 1$$ convolution on the $$1 \times 1 \times 400$$ volume with $$400$$ filters., giving us a $$1 \times 1 \times 400$$ output. The layer after that will also be a $$1 \times 1$$ layer followed by the softmax output.

This forms the basis of sliding windows in object detection.

## Implementing Convolutional sliding windows

Consider a $$14 \times 14 \times 3$$ input put through the following architecture

| Inputs         | No. and Type of Filters | Kernel Size | Output     |
| -------------- | ----------------------- | ----------- | ---------- |
| (14,14,3)      | 16 (CONV)               | 5           | (10,10,16) |
| (10,10,16)     | 1  (MAX POOL)           | 2           | (5,5,16)   |
| (5,5,16)       | 400 (CONV -> FC)        | 5           | 400        |
| 400            | 400 (CONV -> FC)        | 1           | 400        |
| 400            | 4 (CONV -> FC)          | 1           | 4          |

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/sliding_window_model.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    A visualization of the above model. Note the output not containing the softmax we would normally place there.
</div>

Given a test set having $$16 \times 16 \times 3$$, a naive approach would be to use a stride of 2 and run the above model $$4$$ times on the given image to get 4 labels. This would be redundant as there many calculations would be shared between each of the separate windows.

To mitigate this, instead of cropping out each region of the image and feeding it into the network, we just directly run the above architecture on the test image (with the exact same stride and other hyperparameters). The calculations for each layer would look something like the following

| Inputs         | No. and Type of Filters | Kernel Size | Output     |
| -------------- | ----------------------- | ----------- | ---------- |
| (16,16,3)      | 16 (CONV)               | 5           | (12,12,16) |
| (12,12,16)     | 1  (MAX POOL)           | 2           | (6,6,16)   |
| (6,6,16)       | 400 (CONV -> FC)        | 5           | (2,2,400)  |
| (2,2,400)      | 400 (CONV -> FC)        | 1           | (2,2,400)  |
| (2,2,400)      | 4 (CONV -> FC)          | 1           | (2,2,4)    |

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/sliding_window_application.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    A visualization of the above model applied to the test image.
</div>

As we can see, we get a $$2 \times 2 \times 4$$ output at the end. Here, each of the corners can be seen as an fully connected layer output corresponding to a $$14 \times 14$$ section of the original image with a stride of $$2$$. The stride can be changed by varying the kernel size of the Max Pool layer.

Hence, instead of running the forward propagation on four subsets of the input image independently, the computations can be combined into a single forward pass on the whole image.

# Predicting bounding boxes

In the previous method, though we are able to implement sliding windows on our image, the position of the bounding boxes are not necessarily accurate. The code above only implements square bounding boxes with fixed strides, which can be inaccurate in case our image contains an object that doesn't completely fit in our given bounding box.

The solution for this is the YOLO algorithm.

Consider an input $$100 \times 100$$ image, the YOLO algorithm aplits the image up into a grid. Upon each of the grid cells, we apply our classification and localization algorithm. This algorithm has the following labels


$$
y = \begin{bmatrix}
P_c \\ bx \\ by \\ bh \\ bw \\ c_1 \\ c_2 \\ \vdots
\end{bmatrix}
$$

Refer to [Defining the target](#defining-the-target) for a breakdown of this format.

In this case, our target output will have the dimension of $$n \times n \times 8$$ with $$n$$ being the number of grid cells per row/column.

We would have to run our ConvNet model on our inputs such that the output obtained at the end is the required $$n \times n \times 8$$ output volume

Objects are assigned to a grid cell based on its center point's position being contained in the corresponding grid, even if it spans multiple grid cells. Using finer grids helps us avoid multiple objects being captured within a single grid cell.

Since this is convolutionally applied, the computations here are fairly quick, which makes the YOLO algorithm fairly popular, even finding use in real-time object detection.

## Encoding the Bounding Boxes

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/grid_cat.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A grid box applied to the cat image example from before.
</div>

The the example of the cat above. Here, the object has a target label $$P_c$$ of $$1$$ followed by the bounding box values followed by the labeel, which in this case we take as the first label $$c_1 = 1$$. Hence, the vector at hand beomes

$$
y = \begin{bmatrix}
1 \\ bx \\ by \\ bh \\ bw \\ 1 \\ 0
\end{bmatrix}
$$

In the YOLO algorithm, considering the upper left corner of the grid square containing the cat to be $$(0,0)$$ and the lower right point to be $$(1,1)$$, the height is specified as a fraction of the total height of the box, the same being with the width. Effectively, all the calculations here are done relative to the width and height of the grid section.

The values can go greater than one since it is a possibility that the given object spans several cells. This is just one of many ways to specify the bounding box.

# Intersection over union

This is a way to evaluate our object detection algorithm to determine how well it is performing.

Object detection involves localizing an object alongside recognizing it. Consider the image below with the green box being the ground truth bounding box and the red box being the predicted bounding box.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cat_iou.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Here, green is the ground truth bounding boc and red is the predicted bounding box.
</div>

The **intersection over union (IoU)** function computes the intersection over union of the two bounding boxes, which is as follows

$$
\text{IoU} = \frac{\text{Size of the bounding boxes intersection}}{\text{Size of the bounding boxes union}}
$$

Conventionally a the answer is taken to be correct if the IoU value is greater than $$0.5$$. In the case of the bounding boxes overlapping perfectly, the IoU is $$1$$. The threshold value can be changed if we are looking for more accurate bounding boxes.

IoU measures the overlap between the bounding boxes to see how similar they are to each other.

# Non-max supression

Non-max supression is a way to deal with the issue of object detection algorithms detecting of multiple of the same object. It ensures that each object is detected only once.

Considering the image below, despite the center point lying inside a grid square, it is possible for our model to have several positives around the grid containing the center point.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cat_iou.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Here, green is the ground truth bounding box and red is the predicted bounding box.
</div>

Since image classification and localization is being applied to every grid cell, many of them can give a positive for the chance of detecting an object, hence giving us multiple detections. Non-max supression cleans up the detections to give just one detection per object.

Here, it looks at the probabilities of each of the repeated detections, takes the larget one and sets it as the label. It then looks at all the remaining rectangles with high overlap and supresses them.

So, in the above example, we get a $$19 \times 19 \times 8$$ output for which we have the prediction as 