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
        {% include figure.liquid loading="eager" path="assets/img/cat_bounding_box.jpg" class="img-fluid rounded z-depth-1" zoomable=true%}
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
- This is because when we implement the $$5 \times 5$$ filter, it is implemented as $$5 \times 5 \times 16$$ as it is looking at all the channels and must match the outputs.
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

In the previous method, though we are able to implement sliding windows on our image, the position of the bounding boxes are not necessarily accurate.

The code above only implements square bounding boxes with fixed strides, which can be inaccurate in case our image contains an object that doesn't completely fit in our given bounding box. It could be present on the line between two boxes, and it might not even be square in shape, but more rectangular.

The solution for this is the **YOLO algorithm**.

Consider an input $$100 \times 100$$ image with three classes to choose from. The YOLO algorithm aplits the image up into a grid, and the classification and localization algorithm is applied to each of the cells. This algorithm will have the following labels

$$
y = \begin{bmatrix}
P_c \\ bx \\ by \\ bh \\ bw \\ c_1 \\ c_2 \\ c_3
\end{bmatrix}
$$

Any cells not containing an object of interest will be labeled with

$$
y = \begin{bmatrix}
0 \\ ? \\ ? \\ ? \\ ? \\ ? \\ ? \\ ?
\end{bmatrix}
$$

and if an object of interest is actually present in the cells, then we will get

$$
y = \begin{bmatrix}
1 \\ bx \\ by \\ bh \\ bw \\ 0 \\ 1 \\ 0
\end{bmatrix}
$$

Refer to [Defining the target](#defining-the-target) for a breakdown of this format.

In this case, our target output will have the dimension of $$n \times n \times 8$$ with $$n$$ being the number of grid cells per row/column. So, if we had $$3 \times 3$$ grid cells, our target would be $$3 \times 3 \times 8$$.

We would take our ConvNet model, run it on our inputs such that the output obtained at the end is the required $$3 \times 3 \times 8$$ output volume.

The advantage here is that the model will give us precise bounding boxes.

Objects are assigned to a grid cell based on its center point's position being contained in the corresponding grid, even if it spans multiple grid cells. Using finer grids helps us avoid multiple objects being captured within a single grid cell.

Since this is convolutionally applied, the computations here are fairly quick, which makes the YOLO algorithm fairly popular, even finding use in real-time object detection.

## Encoding the Bounding Boxes

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/grid_cat.jpg" class="img-fluid rounded z-depth-1" zoomable=true%}
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

In the YOLO algorithm, considering the upper left corner of the grid square containing the cat to be $$(0,0)$$ and the lower right point to be $$(1,1)$$, the height is specified as a **fraction of the total height of the grid cell**, the same being with the width. Effectively, all the calculations here are done relative to the width and height of the grid section.

The values can go greater than one since it is a possibility that the given object spans several cells. This is just one of many ways to specify the bounding box. Later versions of YOLO go into these.

# Intersection over union

This is a way to evaluate our object detection algorithm to determine how well it is performing.

Object detection involves **localizing** an object alongside recognizing it.

Consider the image below with the green box being the ground truth bounding box and the red box being the predicted bounding box.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cat_iou.jpg" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Here, green is the ground truth bounding box and red is the predicted bounding box.
</div>

The **intersection over union (IoU)** function computes the intersection over union of the two bounding boxes, which is as follows

$$
\text{IoU} = \frac{\text{Bounding box intersection area}}{\text{Bounding box union area}}
$$

Conventionally a the answer is taken to be correct if the IoU value is greater than $$0.5$$.

In the case of the bounding boxes overlapping perfectly, the IoU is $$1$$. The threshold value can be changed if we are looking for more accurate bounding boxes, but it can also lead to longer training times.

## IoU: Sets Intuition

Putting this in terms of sets makes it a bit easier to understand. Refer to the image below.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/set_example.jpg" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

Consider the case where our bounding boxes are in the form of sets that intersect one another. This intersection's value can change as the boxes get closer. The ideal condition for our bounding boxes would be for our intersection to be equal to the union, or as close to it as possible.

Hence, by taking the ratio of the **intersection over union** we can study how close our bounding boxes would be to the ground truth.

# Non-max supression

Non-max supression our object detection algorithms detects the same object only once

Considering the image below, despite the center point lying inside a grid square, it is possible for our model to have several positives around the grid containing the center point.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/non_max_eg1.jpg" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

Since image classification and localization is being applied to every grid cell, many of them can give a positive for the chance of detecting an object, hence giving us multiple detections. Non-max supression gives us one detection per object.

Here, non-max supression looks at the probabilities of each of the repeated detections, takes the largest one and sets it as the label. It then supresses the remaining rectangles.


Looking at it grammatically, we could say that all the probabilities that are **not max** are **supressed**.

## The algorithm

So, in the above example, we get a $$19 \times 19 \times 8$$ output for which we have the prediction is given as.

We consider each of the output predictions to be
$$
y = \begin{bmatrix}
P_c \\ bx \\ by \\ bh \\ bw
\end{bmatrix}
$$

since we are only considering the detection of a cat here.

- The first step: Discard all boxes with $$P_c$$ values less than equal to the threshold set by us.
- Pick the box with the largest $$P_c$$ as the output.
- With the remaining predictions, discard any having an $$\text{IoU} \ge 0.5$$.
- repeat this with boxes that have not yet been processed till each is either output as a prediction or discarded.


For images having multiple objects, the output vector's size increases proportionally, so non-max supression is carried out three times independently.

# Anchor boxes

This mechanism helps detect multiple objects in a single grid cell.

Consider the image below where we have the bounding boxes for a dog and couch.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/doge_anchor.jpg" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

here we see that the midpoints for each are almost at the same place, in the same grid cell. So, for our output vector

$$
y = \begin{bmatrix}
P_c \\ bx \\ by \\ bh \\ bw \\ c_1 \\ c_2
\end{bmatrix}
$$

it is difficult to get two classes in the same box simultaneously.

**Anchor boxes** predefine two different shapes/boxes and associate different predictions with them. So, instead of the vector being as the above, we set it to be

$$
y = \left[\begin{matrix}
P_c \\ bx \\ by \\ bh \\ bw \\ c_1 \\ c_2 \\ \hline P_c \\ bx \\ by \\ bh \\ bw \\ c_1 \\ c_2
\end{matrix}\right]
$$

Where each following set of outputs is related to an anchor box. So if we had bounding boxes like

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/anchor_boxes.jpg" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

The first one would be used to encode the doge and the second would be used to encode the couch.

So,

> #### Previously
> Each object was assigned to a grid cell containing its mid point
{: .block-danger }

> #### Now: With Two Anchor Boxes
> Each object is assigned to a grid cell containing its mid point AND anchor box for the grid cell with the highest IoU
{: .block-tip }

If we have two anchor boxes but multiple objects in a grid cell, **our algorithm will start to struggle**. Here we would have to use something like a tiebreaker.

Another case is if we had two objects in the same grid cell but with similar anchor box shapes, this will also lead to our algorithm struggling.

These are fairly rare to see.

Anchor boxes help specialize out algorithm for specific use cases.

#### Choosing Anchor Boxes

The classical way was to do it by hand in a variety of shapes.

In later YOLO papers, the **K-means** algorithm was used to group together types of objects one would get and use that to select the anchor boxes that would represent the classes we are trying to represent.

# The YOLO Algorithm

Typing up all the concepts we discussed before, we can now study how the YOLO algorithm works.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/yolo_1.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    This image was taken from Coursera.
</div>

Here we are trying to detect
- Pedestrians
- Cars
- Motorcycles

If we use two anchor boxes, we will have the output vector as $$3 \times 3 \times 2 \times 8$$

For our training set, we go through each of the $$9$$ grid cells and form the appropriate target vector $$y$$.

For all the empty grid cells we would get

$$
y = \left[\begin{matrix}
0 \\ ? \\ ? \\ ? \\ ? \\ ? \\ ? \\ ? \\ \hline 0 \\ ? \\ ? \\ ? \\ ? \\ ? \\ ? \\ ?
\end{matrix}\right]
$$

and for cells containing an object we would get

$$
y = \left[\begin{matrix}
0 \\ ? \\ ? \\ ? \\ ? \\ ? \\ ? \\ ? \\ \hline 1 \\ b_x \\ b_y \\ b_h \\ b_w \\ 0 \\ 1 \\ 0
\end{matrix}\right]
$$

So, we basically train a ConvNet that inputs an image of, say, $$100 \times 100 \times 3$$ to an output of $$3 \times 3 \times 2 \times 8$$

## How we make predictions

Given an image with a grid, we get the $$3 \times 3 \times 2 \times 8$$ output for each of the grid cells, giving us relevant outputs for each.

Finally, we run the network through non-max supression:
- If using two anchor boxes, for each grid cell, we get two predicted bounding boxes with varying $$P_c$$.
- Get rid of the low probability predictions.
- When trying to detect 3 classes, for each class, uses non-max supression to get the final predictions.

This is one of the most effective object detection algorithms encompassing some of the best ideas across CV literature related to object detection

# Region Proposal: R-CNN

Looking back at sliding windows, we would take a small window thoughout an image and train a classifier on each of them.

The main downside is the training time and reclassifying regions where there is no object.

Here is where R-CNNs, or regions with CNNs come in. Here it tries to pich just a few windows where it would run the CNN classifier.

This is done with the help of a **segmentation algorithm** to figure out what could be objects and run the classifier around just those blobs

R-CNNs are still pretty slow, so faster algorithms have been developed:
- **R-CNN:** Propose regions, classify on them one by one, output label + bounding box.
- **Fast R-CNN:** Propose regions, Use convolutional implementation of sliding windows to classify proposed regions. (Drawback: the clustering step to propose regions is slow)
- **Faster R-CNN:** Use CNN to propose regions.

These are all still a bit slower than the YOLO algorithm.

# U-Net (Semantic Segmentation)

All the examples till now have dealt with objext recognition and detection. Now we will look at **semantic segmentation**.

Segmentation outlines the object to know which pixels _exactly_ belong to the object and which don't rather than using bounding boxes.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/semseg_eg.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>
<div class="caption">
    Source: https://medium.com/hackabit/semantic-segmentation-8f2900eff5c8
</div>

One of the uses of semantic segmentation is in self driving cars to know which exact pixels are safe to drive on.

Other uses include chest x-ray studies, brain MRI scans

## What Semantic segmentation does

Consider the given image where we want to segment out the woman who's angry at something (I wonder what).

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/angry_woman.jpg" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

Here we would have two class labels, 1 for the woman and 0 for nothing. Below, I have only marked the woman since marking the whole image wouldn't be worth the effort.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/angry_woman_1.jpg" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

The segmentation algorithm outputs 1 or 0 for every pixel in the image if it is part of the object we are interested in.

Similar to previous examples, we can use multiple labels for different classes of pixels.

So, our network will have to generate a matrix of labels.

## Building a semantic segmentation network

Consider the CNN below

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/CNN_UNet.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

We now want to make the output of this a whole matrix of class labels.

What we do is eliminate the last few fully connected layers.

Now, as we go down a network, the size of the image gradually goes down. We now need to blow the size of our image back up.

The architecture will represent something like below where, as we go deeper, the height and width go back up while the number of channels decreases.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/CNN_UNet_2.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

This blowing up mechanism is done with the help of **transpose convolutions**.

## Transpose Convolutions

This is the main part of the U-Net architecture responsible for blowing up lower dimensional convolutions to higher dimensional ones.

Consider the example we are already familiar with, the normal convolution

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/RGBConv1.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

Where a $$6 \times 6 \times 3$$ image is convolved with five $$3 \times 3 \times 3$$ filters to get a $$4 \times 4 \times 5$$ output.

A transpose convolution is different from this.

Here we can take a smaller set of activations, say $$2 \times 2$$, convolve it with a $$3 \times 3$$ filter to get a $$4 \times 4$$ output.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/transConv1.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

Consider the following

$$
\begin{bmatrix}2 & 1 \\ 3 & 2 \end{bmatrix} \xrightarrow{\begin{bmatrix}1 & 2 & 1 \\ 2 & 0 & 1 \\ 0 & 2 & 1 \end{bmatrix}} \begin{bmatrix}* & * & * & * \\ * & * & * & * \\ * & * & * & * \\ * & * & * & *\end{bmatrix}
$$

Here we use a padding $$p=1$$ on the output and a stride of $$s=2$$.

$$
\begin{bmatrix}2 & 1 \\ 3 & 2 \end{bmatrix}
\xrightarrow{\begin{bmatrix}1 & 2 & 1 \\ 2 & 0 & 1 \\ 0 & 2 & 1 \end{bmatrix}}
\begin{bmatrix}
\colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } \\
\colorbox{grey}{ } & * & * & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{ } & * & * & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{ } & * & * & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{ } & * & * & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ }& \colorbox{grey}{ }
\end{bmatrix}
$$

The main difference here is that, in contrast to the filter being placed on the input, multiplied and summed up; we place the filter on the output

So, taking the number $$2$$ and multiplying it with every value in the filter, we get the output of the upper left corner of $$3 \times 3$$ pixels.

$$
\begin{bmatrix}\colorbox{red}{2} & 1 \\ 3 & 2 \end{bmatrix}
\xrightarrow{\begin{bmatrix}1^2 & 2^2 & 1^2 \\ 2^2 & 0^2 & 1^2 \\ 0^2 & 2^2 & 1^2 \end{bmatrix}}
\begin{bmatrix}
\colorbox{red}{ } & \colorbox{red}{ } & \colorbox{red}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } \\
\colorbox{red}{ } & \colorbox{red}{*} & \colorbox{red}{*} & * & * & \colorbox{grey}{ } \\
\colorbox{red}{ } & \colorbox{red}{*} & \colorbox{red}{*} & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{ } & * & * & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{ } & * & * & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ }& \colorbox{grey}{ }
\end{bmatrix}
$$

Here the padding area is ignored and only the values for the $$2 \times 2$$ region is filled with the convolution region as shown

$$
\begin{bmatrix}\colorbox{red}{2} & 1 \\ 3 & 2 \end{bmatrix}
\xrightarrow{\begin{bmatrix}1^2 & 2^2 & 1^2 \\ 2^2 & \colorbox{red}{0^2} & \colorbox{red}{1^2} \\ 0^2 & \colorbox{red}{2^2} & \colorbox{red}{1^2} \end{bmatrix}}
\begin{bmatrix}
\colorbox{grey}{X} & \colorbox{grey}{X} & \colorbox{grey}{X} & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } \\
\colorbox{grey}{X} & \colorbox{red}{0} & \colorbox{red}{2} & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{X} & \colorbox{red}{4} & \colorbox{red}{2} & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{ } & * & * & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{ } & * & * & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ }& \colorbox{grey}{ }
\end{bmatrix}
$$

Similarly, for the second element, we face an overlap with the values from the first part. In this case, we add up the values

$$
\begin{bmatrix}2 & \colorbox{blue}1 \\ 3 & 2 \end{bmatrix}
\xrightarrow{\begin{bmatrix}1^1 & 2^1 & 1^1 \\ \colorbox{blue}{2^1} & \colorbox{blue}{0^1} & \colorbox{blue}{1^1} \\ \colorbox{blue}{0^1} & \colorbox{blue}{2^1} & \colorbox{blue}{1^1} \end{bmatrix}}
\begin{bmatrix}
\colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{X} & \colorbox{grey}{X} & \colorbox{grey}{X} & \colorbox{grey}{ } \\
\colorbox{grey}{ } & 0 & \colorbox{red}{2}+\colorbox{blue}{2} & \colorbox{blue}{0} & \colorbox{blue}{1} & \colorbox{grey}{ } \\
\colorbox{grey}{ } & 4 & \colorbox{red}{2}+\colorbox{blue}{0} & \colorbox{blue}{2} & \colorbox{blue}{1} & \colorbox{grey}{ } \\
\colorbox{grey}{ } & * & * & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{ } & * & * & * & * & \colorbox{grey}{ } \\
\colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ }& \colorbox{grey}{ }
\end{bmatrix}
$$

Repeating this for the remaining values following the same process and adding up the values obtained, we get the final result as follows

$$
\begin{bmatrix}2 & 1 \\ 3 & 2 \end{bmatrix}
\xrightarrow{\begin{bmatrix}1 & 2 & 1 \\ 2 & 0 & 1 \\ 0 & 2 & 1 \end{bmatrix}}
\begin{bmatrix}
\colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } \\
\colorbox{grey}{ } & 0 & 4 & 0 & 1 & \colorbox{grey}{ } \\
\colorbox{grey}{ } & 10 & 7 & 6 & 3 & \colorbox{grey}{ } \\
\colorbox{grey}{ } & 0 & 7 & 0 & 2 & \colorbox{grey}{ } \\
\colorbox{grey}{ } & 6 & 3 & 4 & 2 & \colorbox{grey}{ } \\
\colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ } & \colorbox{grey}{ }& \colorbox{grey}{ }
\end{bmatrix}
$$

## U-Net Intuition

The architecture is rougly as follows

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/CNN_UNet_2.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

The first part of the network is composed of normal convolutions which compresses the image, and the second half uses transpose convolutions to bring the size back up to the original size of the image.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/UNET_0.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

The main change made to the above that converts it into a U-Net is skip connections add from earlier blocks to later blocks. The reasons for this are:

- We need high level spatial information which we can get from the **previous layer** from which the NN may have figured out that there is an object at a certain part of the image
- The original input here has a very low spatial resolution. We need detailed, fine-grained spatial information. The **skip connection** allows us to send the high-resolution low level feature information to the corresponding layer in the network

## U-Net Architecture

The U-Net is one of the most important models today. The image below shows an example of a U-Net from the original paper.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/UNET.png" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

In the above image, instead of visualizing the image and the subsequent layers in 3D, they are instead represented with lines of varying width. This makes it a bit easier to visualize stuff since we already have an idea of how our inputs and outputs look and how the convolutions act upon them.

The given image is pretty self explanatory where:
- Each black arrow represents a convolution operation on the inputs
- Red arrows represent pooling layers that reduce the dimension of the inputs
- Green arrows represent transpose convolutions
- White arrows represent skip connections sending information from the corresponding lower layers.

We obtain a $$h \times w \times n_{\text{classes}}$$ output corresponding to the dimensions of our input and each layer corresponding to bounding boxes for our classes.

# Conclusion

This, rather short blog covers a few basic object detection concepts along with two fairly relevant object detection models: YOLO and U-Nets. Both these models find several use cases in the modern day, especially YOLO with continuous developments happening on the daily. I would be interested to implement these in various use cases as well as build them up from scratch to see how they work.