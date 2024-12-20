---
layout: post
title: Deep Learning - An Introduction
date: 2024-09-17 16:40:16
description: "Topics covered: Introduction to Neural Networks and the underlying mathematics behind the training process for shallow and deep Artificial Neural Networks."
tags: machine-learning deep-learning
categories: deep-learning
toc:
  sidebar: left
bibliography: 2024-06-30-dl-1.bib
related_publications: true
---

This is the first of a series of posts that compile my reports for a Deep Learning independent study taken up at the University of Massachusetts Dartmouth.

This study is loosely based on the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) offered by [deeplearning.ai](https://www.deeplearning.ai/) with more content included from outside sources where necessary. A lot of the visualizations here are hand-drawn or taken directly from Coursera.

The focus here is to start from the foundations to help understand the capabilities, challenges and consequences of deep learning, gradually moving on to more complex topics and applications of these concepts. The main goal is to gain insights on the inner workings of the concepts behind cutting-edge AI models used in various fields as well as hands-on experience using these models. Equal amounts of importance will be given to the underlying logic and mathematics behind various deep learning models and their practical applications.

This post covers the basics of deep learning and neural networks with a focus on the underlying computation.

<hr>

# Introduction

In the world of data science {% cite coursera2024 %}, we usually see mentions of statistics, machine learning (ML), artificial intelligence (AI), and deep learning several times. Though machine learning and deep learning are both subfields of AI and both terms are used interchangeably, they have their own meanings.

**Machine learning** algorithms are relatively simple models designed to adapt on input data and either find patterns or make decisions on newly provided data.

**Deep learning** on the other hand, is a branch of machine learning that involves training expansive _artificial neural networks (ANNs)_. An ANN consists of layers of interconnected nodes called _neurons_ that, together, help process and learn patterns from the input.

We often picture AI, deep learning and machine learning as a Venn diagram with AI taking up the outermost layer, machine learning being the intermediate layer, and the innermost layer being deep learning. A more detailed visualization of this can be seen below.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/AI_domains.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Artificial Intelligence and its Domains
</div>

Deep learning has not only transformed how we search for content on the internet, but has also found its use in domains such as:
- **Image Recognition**: Image recognition is one of the most common uses of deep learning, where a model is capable of identifying objects or features in image and video data. This forms the groundwork for several fields such as facial recognition, image classification, automated driving, robotics and several other fields. This domain also has a great amount of overlap in the field of healthcare in terms of medical imaging.
- **Object Detection**: Object detection is a technique similar to image recognition. Both these techniques are often paired together when being used practically. Object detection is used on a higher level to detect instances and the positions of objects in an image or video, while image recognition is capable of assigning an identity to objects from these sources. E.g. Object detection would detect whether an image contains cats in it along with the number of cats and their positions. Image recognition would be capable of telling us what color or breed of cat is present in the given image.
- **Natural Language Processing**: This domain is related to creating a way for computers to understand natural human languages in everyday speech. This could be in several forms such as speech, writing or even sign language. Some of the common techniques that show up in this field are sentiment analysis, keyword extraction, translation, text summarization, text generation, etc.
- **Generative AI**: This domain involves generating various types of context such as text, images and audio, and is one of the most popular applications of deep learning as of the writing of this blog. GenAI has found various uses in boosting human productivity, such as in content creation, data analysis and research. Large Language Models (LLMs) are the most popular applications of GenAI, with active research and development being done in the domain. Some popular LLMs are Llama by Meta, ChatGPT and GPT-4 by OpenAI, and Gemini by Google.

# Neural Networks

The basic idea of deep learning is the training of **Neural Networks**. The best way to understand what a neural network is, is to look at a real-life example.

##### Scenario 1

Consider a housing price prediction model where the data at hand consists of the square-foot size of the houses and their prices. We plot these points out taking the square-foot size on the X axis and price on the Y axis.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/housing_example_1.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    All the instances of houses plotted by Price vs Square-foot size.
</div>

Using basic machine learning, the next logical step would be to create a regression or any other suitable model to fit a line or curve that best describes the general trend of the price of the houses with relation to their size. In this case, let us take linear regression as the initial model of choice.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/housing_example_2.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The linear regression line plotted to fit the points. Note the line taking up negative price values.
</div>

Now, we know that the price of a house can never be negative. That would imply that the buyer is being paid a sum of money along with receiving a house, which does not make sense realistically. Pure linear regression takes into account both negative and positive values, which is why we can rule it out. Instead, we truncate our fitting line at $$ y=0 $$, giving us the following that predicts the price as a function of size.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/housing_example_3.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The modified linear regression line with truncation (a.k.a ReLU).
</div>

This resulting function is known as a **Rectified Linear Unit** or **ReLU** for short, and is widely used in the development of deep learning models.

##### Scenario 2

Now, suppose we have more features to work with in addition to the size of the house, such as the number of _bedrooms_, _zip code_ and _wealth_ of the neighborhood. In this scenario we can make the following observations:
- The _size_ and _number of bedrooms_ gives us an idea of whether a house can accommodate for the _family size_.
- The _zip code_ tells us how _walkable_ the locality of the house is.
- The _zip code_ along with the _wealth_ of the neighborhood can give us an idea of the _quality of schools_ in the area.

All of this information can be visualized in the form of a neural network as shown.

<div class="row justify-content-center mt-3">
    <div class="col-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/housing_example_nn_1.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The circles in the image here represent **nodes** which can contain a ReLU or any other function. The arrows here represent the flow of information between the input and nodes.

All the observations made above can be seen as functions that take in the information to give the corresponding outputs for family size, walkability and school quality. In a similar fashion, all of these can together be taken in as inputs for yet another node which gives us an estimate for the price of the given house.

This example gives us a look into how people can go about deciding how much they would be willing to spend on a house.

## A real-world Neural Network

In a real-world implementation of a neural network, the inputs are taken as a vector $$X$$ with the target (price) taken as $$y$$ which we aim to predict.

We give the neural network the $$X$$ values along with the target $$y$$ as the training set. As opposed to the previous scenarios, the function of the intermediate layers are figured out by the neural network by itself.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/actual_nn.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A real-world implementation of a single layer neural network.
</div>

A practical neural network would look like the image above. The circles here are called **hidden units** which are arranged in layers known as **hidden layers**.

Each node here takes inputs from all of the input features. We do not have control over what each node does. The neural network itself makes the decision on what each node is supposed to do.

In addition, a practical neural network will seldom have just one neuron or a single layer. Instead a collection of several neurons receiving differing inputs would be used, which collaborate to give an accurate and complex prediction.

In this case, the layer  is known as a **densely connected layer** as every input feature of a given layer is connected to every other node in its previous and next layer.

The main advantage of neural networks is that with a large enough number of input features along with the target, and enough training examples for them, neural networks are effective at figuring out approximate functions to map the inputs to the outputs.

## Supervised Learning with Neural Networks

Supervised learning makes use of labelled datasets where we have both the features $$X$$ and the target $$y$$ available to us. The goal here is to learn a function that would best form a relation ship between them.

Neural networks are widely used for supervised learning tasks due to their adaptability and capability of creating complex representations.

A popular implementation of neural networks is in Natural Language Processing, where they are used in tasks such as sentiment analysis, machine translation and speech synthesis.

Computer Vision is another implementation of neural networks where machines are trained to interpret images and make decisions based on the same. Some popular computer vision tasks are object detection, image recognition and facial recognition.

Neural networks also find use in the domain of healthcare {% cite DLNIH %} where they are employed in fields such as biomedical informatics, genomics and public health to gain insights from high dimensional and complex data. Several challenges and opportunities are faced in this field, especially when developing data-driven tools for knowledge discovery and predictive analysis. Neural networks find their use here due to them being capable of creating models to more accurately represent the complex relationships in the data and create meaningful abstractions from it.

### Sequential Data

A lot of modern implementations of deep learning involves working with **sequential data**. Sequential data refers to a series of data points whose order of occurrence is of importance. Examples of these are time series data, audio signals, stock price data, natural language sentences, gene sequences and weather data. In all these cases, the points in the dataset are dependent on their position in relation to other points present in the dataset.

Traditional neural networks are not capable of handling sequential data well as they are incapable of taking into consideration of the sequence in which the data is fed into them. Instead specialized neural networks have been developed to handle these cases, some of which are:
- **Recurrent Neural Networks (RNNs)**: RNNs are designed to work with sequential data. This is attributed to their "memory" mechanism where they can take in information from prior inputs to influence the outcome of the current input and output. As opposed to traditional neural networks which assume the inputs to be independent of each other, RNN outputs depend on the prior elements in the sequence. This makes them useful in fields such as speech recognition and language modelling.
- **Long Short-Term Memory (LSTM) Networks**: LSTMs {% cite LSTM %} are a variant of RNNs developed as a solution to several problems faced by RNNs such as the vanishing gradient problem. These models can memorize patterns over long sequences, making them useful for time series predictions with longer patterns such as machine translation, speech recognition and video analysis.
- **Gated Recurrent Units (GRUs)**: GRUs, similar to LSTMs also address the problem of short-term memory in RNNs and simplify the overall architecture of LSTMs.
- **Transformers and Attention Mechanisms**: These architectures while not being recurrent, handle sequences in the input data by focusing on input elements that are considered important. They prioritize focusing on relevant information, leading to an enhancement of the overall performance of the model.

### Non-Sequential Data

Non-sequential data is more straightforward to deal with as the inputs are independent of one another. Such information can be modelled with the use of **FeedForward Neural Networks (FNNs)** or **Multi-layer Perceptrons (MLPs)**. These networks are quintessential to deep learning and aim to approximate a function mapping its inputs to the outputs. They consist of an input layer, several hidden layers and an output layer. In these networks the neurons in a given layer are fully connected with the neurons in the subsequent layer, hence enabling the learning and transformation of features. Such networks are used for tasks such as regression and classification and more complex tasks in combination with other techniques.

In the subsequent sections we will dive deeper into how these networks work.

## Neural Networks: The Basics

**Deep feedforward networks** {% cite Goodfellow-et-al-2016 %} are the most basic of deep learning models with the main goal of approximating a function. Feedforward networks get their name from the flow of information through the function being evaluated from the inputs $$X$$, through intermediate computations towards the output $$y$$. These intermediate computations eventually define the function we are approximating.


<div class="row justify-content-center mt-3">
    <div class="col-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/DNN1.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A typical feedforward network.
</div>

Neural networks, as can be seen form the image, compose different functions together. A good example given by Goodfellow's book (previously cited) takes three functions $$f^{(1)}, f^{(2)}, f^{(3)}$$ which are connected in a chain to form a composite function of the form $$f(x) = f^{(3)}(f^{(2)}(f^{(1)}(x)))$$, where the numbers in the superscript correspond to different **layers** of nodes. This is what gives us the term **networks** to describe these models.

The very first layer of the feedforward network consists of the inputs, aptly giving it the name: the **input layer** with the final layer of the network being called the **output layer**. The layers that lie between the input and output layers are called **hidden layers**.

Training a neural network has the aim of finding an approximate function, say $$f(x)$$ that can match an unknown function, say $$f^{*}(x)$$. The training process of a neural network aims to drive the function $$f(x)$$ to match $$f^{*}(x)$$ as close as possible. Each training example gives us noisy approximate examples of $$f^{*}(x)$$, hence specifying directly what a layer must do at the given point. The training algorithm must decide how the layers are used together towards producing $$f$$ such that it is as close as possible to $$f^{*}$$.

The most common analogy for naming these networks as **neural networks** is that they take inspiration from neuroscience, owing to their resemblance to human neurons. A human neuron receives electrical signals from other neurons, does a simple computation and if fired, it sends a pulse of electricity down the axon to other neurons. Similarly, each layer of a neural network is vector valued, with each vector element representing a neuron and the number of elements representing the **width** of the model.

However, it is still up to question how neurons actually work as they are much more complex than the given explanation. We are not sure if a neuron uses an algorithm or any other different learning principle. Hence, deep learning can learn complex function mappings fairly easily, but does not directly compare to a neuron.

The easiest way to understand how neural networks work is through examples, one of which we will have a look at now.

## Notation

Before we go on to building our network, here are some notations to keep in mind that we will be using.
- A single training example is $$(x,y)$$ where $$x \in \mathbb{R}^{n_{x}}$$ and $$y \in \{0,1\}$$
- Training set is denoted by $$m$$ or $$m_{train}$$ : $$\{(x^{(1)},y^{(1)}), (x^{(2)},y^{(2)}), \dots, (x^{(m)},y^{(m)})\}$$
- Testing set is denoted by $$m_{test}$$
- The training set as a column matrix is denoted as
$$X = \begin{bmatrix} \vdots & \vdots & & \vdots\\ x^{(1)} & x^{(2)} & \dots & x^{(m)}\\ \vdots & \vdots & & \vdots\\ \end{bmatrix} $$, with the number of columns being $$m$$ and the number of training examples/rows/height is $$n_{x}$$. $$X \in \mathbb{R}^{n_{x} \times m}$$
- The output labels are also in the form of a row vector corresponding to each column of $$X$$ such that $$Y \in \mathbb{R}^{1 \times m}$$.
- Different layers of the network are denoted with superscript square brackets. For instance, $$z^{[1]}$$ and $$a^{[1]}$$ correspond to computations on the first layer, $$z^{[2]}$$ and $$a^{[2]}$$ correspond to computations on the second layer, and so on.

## Logistic Regression

In the modern day, we use a lot of different types of artificial neurons, the most common of which is the **sigmoid neuron**, which is used for binary classification via **logistic regression**.

The formula of the sigmoid function is given by the formula

$$f = \frac{1}{1 + e^{-\sum\limits\beta_nx_n}}$$

where $$\beta_n$$ is a **weight** and $$x_n$$ is a **feature**.

Consider the example of an image that we have to classify as either a cat (1) or not a cat (0). The image is in the form of 3 matrices of red, green and blue channels. So, for a $$64 \times 64$$ pixel image, we have 3 $$64 \times 64$$ matrices corresponding to rgb pixel intensities for the image.

To convert these into a feature vector, we unroll the matrices into an input feature $$x$$ as

$$\begin{bmatrix} 255\\ 231\\ \vdots\\ 255\\ 134\\ \end{bmatrix}$$

which is a combination of all the rgb pixel intensity values into a single vector.

For a $$64 \times 64$$ image, the dimension of this vector will be $$64 \times 64 \times 3$$ as it is the total number of elements we have which in this case is 12288. We use $$n_{x}= 12288$$ to represent **dimension of input features** $$x$$ and sometimes for brevity we just use $$n$$.

The goal here is to use an input image represented by the feature vector $$x$$ to predict whether the corresponding label $$y$$ is 1 or 0.

Now, given the input images $$X$$ we want to predict $$\hat{y} = P(y=1 \mid x)$$.

Here the **parameters** we work with are:
- The **weights**, $$w \in \mathbb{R}^{n_{x}}$$, which are vectors and
- The **biases**, $$b \in \mathbb{R}$$, which are scalars.

This, is the basis of a **linear model** given by

$$\hat{y} = w^{T}x + b$$

The dot product of $$w^T$$ and $$x$$ helps us evaluate multiple classifiers parallelly where each $$x$$ would have a different value of $$w$$.

The drawback of a linear classifier is the fact that it has a range of values in $$(-\infty, +\infty)$$, giving us output values that have a large amount of variations as opposed to a simple yes or no equivalent.

This is where logistic regression comes in. The logistic function

$$\hat{y} = \sigma(\underbrace{w^{T}x + b}_{z}) = \frac{1}{1 + e^{-z}}$$

clamps the value of the linear function within the range $$[0,1]$$ such that
- for very large values of $$z$$, $$e^{-z} \rightarrow 0$$, hence giving us $$\sigma(z) \approx 1$$
- for very large negative values of $$z$$, $$e^{-z} \rightarrow \infty$$, hence giving us $$\sigma(z) \approx 0$$

Our goal here is to learn the parameters $$w$$ and $$b$$ such that the given function $$\hat{y}$$ is a good estimate of $$P(y=1 \mid x)$$. This is done with the help of the **cost function**.

## Cost Functions

We already know loss functions to be the measure the performance of an individual training example. The **cost function** is the average of the loss functions over the entire training set. Cost functions measure how well the parameters are doing over the entire set. The main goal is to minimize the cost function with the help of the appropriate weights and biases.

When designing a neural network, the choice of cost function we use holds a great deal of importance. Cost functions in the case of neural networks are the same as those of other models parametric models.

The most common parametric models we use define a distribution, which work on the idea of maximum likelihood, hence making use of cross-entropy between the training data and predictions as the  cost function (such as is in the case of logistic regression or softmax).

In practice, when training a neural network, the total cost function would be a combination of the primary cost function with a regularization term.

### Logistic Regression Cost Function

Consider the data points $$\{(x^{(1)},y^{(1)}), (x^{(2)},y^{(2)}), \dots, (x^{(m)},y^{(m)})\}$$, where we want to find $$\hat{y}^{(i)} \approx y^{((i))}$$. the function being taken here is the sigmoid function given by $$\hat{y} = \sigma(w^{T}x + b)$$ or

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

by finding the parameters $$w$$ & $$b$$.

We judge the performance of our model by using an **error/loss function** denoted by $$L(\hat{y}, y)$$. The loss function here helps us understand how well our current approximating function's output $$\hat{y}$$ is doing in comparison to the ground truth $$y$$.

In most cases, the **square error** or **half of the square error** would be taken into consideration for a linear problem. In the case of Logistic Regression, the optimization problem would be non-convex with multiple local minima if we made use of this loss function.

Hence, instead we make used of the loss function for Logistic regression, given by

$$L(\hat{y}, y) = -(y\log{\hat{y}} + (1-y)\log{
(1-\hat{y})})$$

For squared error, we normally want the value to be as small as possible. Similarly for logistic regression, we would want this loss function to be as small as possible.

The mechanism of this loss function is rather interesting to look at. It is a version of **Cross-Entropy Loss** which is used to calculate the loss for the softmax function, which we will look into later on. Its working is as follows
- Take $$y=1$$, then we get $$L(\hat{y}, y) = -\log{\hat{y}}$$, which should be as small as possible.
	- This means that $$\log{\hat{y}}$$ should be as large as possible.
	- And hence $$\hat{y}$$ should be large but never bigger than 1.
- On the flip side, if $$y=0$$, then we would get $$L(\hat{y}, y) = -\log{(1-\hat{y})}$$, which should be as small as possible.
	- $$\log{(1-\hat{y})}$$ should be as large as possible.
	- hence $$\hat{y}$$ should be as small as possible but never lesser than 0.

If we are working with the complete training set, instead of a loss function, we would make use of the **cost function** given by

$$J(w,b) = \frac{1}{m}\sum\limits_{i=1}^{n} L(\hat{y}^{(i)}, y^{(i)}) = -\frac{1}{m}\sum\limits_{i=1}^{n} y^{(i)}\log{\hat{y}^{(i)}} + (1-y^{(i)})\log{(1-\hat{y}^{(i)})}$$

where we want to find $$w$$ and $$b$$ such that it minimizes the overall cost function $$J(w,b)$$. We will be looking into the mechanism for this now.

## Gradient Descent

As opposed to most models we usually see in machine learning, neural networks are non-linear, leading to the loss functions being non-convex. Hence, neural networks need to be trained iteratively using gradient based optimizers to drive the cost function low as possible.

Gradient descent is one of the most common algorithms used in machine learning, and in this case, it also finds its use in learning the parameters $$w$$ and $$b$$ on our training set.

When training neural networks using gradient descent, it is important that all the weights are initialized to small random values with the biases being either zero or small positive values. We will look into the reason for this later in this article.

Gradient descent can be visualized as shown

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Gradient Descent.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The cost function here is seen as a surface above the horizontal axes of $$w$$ and $$b$$ with the height giving us the value of $$J(w,b)$$ at a certain point.

We aim to find the value of $$w$$ and $$b$$ such that the cost of our resultant model corresponds with the minimum of this curve. In this case, the cost function we see is a convex function due to which it can be seen as a big bowl with a single optimum point.

To find a good value for these parameters, we first initialize the $$w$$ and $$b$$ values to an initial value. The gradient descent algorithm takes a step in the steepest downhill direction, trying to find its way down to the minima as quickly as possible. This step is repeated till our function reaches the global optimum or a value close to it.

Visualizing this, consider a two-dimensional representation for the cost function, ignoring the bias $$b$$ for the sake of simplicity while we are at it.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/GD1.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The cost function in 2D
</div>

The following steps are taken to reach the minima:
- $$w := w - \alpha \frac{dJ(w)}{dw}$$ : updating the value of $$w$$. This is known as the **inner loop**
- Repeat the above till it converges to a global minimum.

Here, $$\alpha$$ is the **learning rate**, which controls how large or small the gradient descent steps taken are.

The value $$\frac{dJ(w)}{dw}$$ is a derivative that depicts the update of the change we want to see in the parameter $$w$$, and which direction it goes in. In code we write this as ```dw```.

Now, suppose we have an initial point for the cost function $$J(w)$$.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/GD2.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Cost function with the initial cost.
</div>

Here, the derivative is the slope of the cost function at a point. It is the height divided by the width of the lower triangle, or the tangent to the curve at the given point.

In this case, we see that the derivative is positive, leading to $$w$$ getting updated as $$w - \alpha \frac{dJ(w)}{dw}$$, hence taking the cost a step to the left.

Continuing this, our algorithm can be made to slowly decrease the parameter till we reach the minima of $$J$$.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/GD3.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Gradient descent updating the cost such that it approaches the minimum.
</div>

For the case of the weight $$w$$ leading the cost to be on the left hand side of the minimum, we would see a similar trend with the only difference being that $$w$$ is increased as $$w + \alpha \frac{dJ(w)}{dw}$$ (since $$\frac{dJ(w)}{dw} < 0$$) till we reach the minimum value of $$J$$.

In the case of logistic regression, we have the cost function inner loop as

$$w = w - \alpha \frac{dJ(w,b)}{dw}$$

$$b = b - \alpha \frac{dJ(w,b)}{db}$$

Here, the above derivatives are partial derivatives, which in code are represented as `dw` and `db`.

## Computation Graphs

Computation graphs are a way for us to visualize mathematical expressions in a descriptive and visual manner. These are extremely useful in understanding how deep learning models work

The computations in a neural network are of two forms:
- **Forward pass/propagation** where the _output_ of the neural network is computed.
- **Backward pass/propagation**, which is used to calculate the _gradients/derivatives_.

Here, they find use in describing both the forward and the backward propagation algorithm in a precise manner.

In a computation graph, each node represents a variable. Variables can be formed from an operation or set of operations (a.k.a a function) consisting of one or more variables. An operation can only return a single output variable.

## Forward Propagation

The process of computing the output of a neural network is done through **forward propagation**. This process is similar to logistic regression but repeated for each node in the hidden layer.

Neural networks, as discussed before, can be seen as multiple functions stacked together as multiple layers of computations. Each node in the network performs two actions:
- Calculating the linear value: $$z$$
- Computing a non-linear activation function $$a$$ using the $$z$$ value.

### Forward Propagation: Explained with Computation Graphs

Taking a simple example of computing a function

$$J(a,b,c) = 3(a + bc)$$

The computation of this function can be split into 3 different parts:
- computing $$u = bc$$
- computing $$v = a + u$$
- computing $$J = 3v$$

In a computation graph, these steps are visualized as follows

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/CG1.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

As we can see, computation graphs are useful when there is some output variable ($$J$$ in this case) to optimize. For logistic regression, $$J$$ is the cost function that we are trying to minimize.

In this case, for a left to right pass, we compute the value of $$J$$ and, in order to compute the derivatives, we see a right to left pass in the opposite direction of the above arrows.

Summing it up, computation graphs organize our left-to-right and right-to-left computations with arrows to give us an understanding of the mathematical model of our network.

## Backward Propagation

**Backward propagation** also known as **backprop** is an algorithm used for supervised learning mainly used in the training of artificial neural networks. It is a gradient descent method used to minimize the error in the predictions of our network, which it does so by adjusting the weights and biases.

Backpropagation allows for the backward flow of information throughout the network, helping us compute the gradient.

The process of backward propagation is as follows:
- **Forward Pass**: This is the first step in the backprop process where the input data is passed through the layers of the network, going through all the hidden layers to get an output prediction.
- **Loss Calculation**: Upon receiving the output predictions, we can compare it with the ground truth by taking the difference between them via a loss function.
- **Backward Pass**: This is the main backpropagation step where the errors are calculated, passed through the network, and eventually used to update the weights and biases through gradient descent. Here:
    - The error obtained is used to calculate the partial derivatives (gradients) of the loss/cost function with respect to the weights $$w$$ and biases $$b$$.
- **Updating the Weights and Biases**: The gradients calculated are propagated backwards through the network starting from the output, throughout the hidden layers. This is usually done by gradient descent, or other similar optimization parameters.
    - As the errors are being propagated, the parameters at each node are updated according to how much they contribute towards the total loss. The gradients calculated tell us how much a change in the given parameter contribute towards the total loss.
    - The adjusting of the parameters is done in the opposite direction of gradient so as to reduce the error obtained.
    - A **learning rate** hyperparameter ($$\alpha$$) is used to control how much the parameters are being updated by at each step of backpropagation.
- **Iteration**: The steps above are repeated until we either the error stops reducing significantly or the loop reaches the maximum number of predefined iterations.

### Importance of Backpropagation
- Backpropagation is efficient due to its calculating the gradient, directly informing the network of how the parameters should be adjusted to reduce the error.
- Backpropagation can be applied to any differentiable loss function and network architecture. This is known as universality.

### Challenges faced in Backpropagation
Some common challenges faced during the backpropagation process are:
- **Vanishing and Exploding gradients**: These are some of the most common problems faced during backpropagation. These occur when the gradients of the weights and biases become either too small or too large as they propagate through the network, making the learning process slow or unstable respectively. These are caused by various factors such as the choice of activation function, parameter initialization or depth of the network.
- **Local Minima**: This is another challenge faced during backprop, where the model gets stuck in a local minimum or saddle point of the loss function. These prevent the network from reaching the optimal solution. These can be combated by using techniques such as momentum, AdamProp, or adaptive learning rates.
- **Computational complexity**: Backpropagation can turn out to be computationally expensive or lead to issues with memory limitations due to the need for storing and updating values and gradients of all the parameters and activations. This limits how complex we can make our network and the quality of the data we can use for training. For this we use techniques like mini-batch training, pruning and quantization.
- **Overfitting and Underfitting**: This is an issue that is faced both in the sphere of deep learning and traditional machine learning. Both these problems lead to poor performance and accuracy of the network. To prevent this, some techniques we can use are regularization, data augmentation, and dropout.

### Backward Propagation: Explained with Computation graphs

In a neural network, during **forward propagation** the inputs $$x$$ provide the initial information that propagates through the hidden units to produce the output $$\hat{y}$$, hence giving us a cost function $$J$$ to work with.

Taking the previous example, we now plug in some values and take the case that we want to compute the derivative of $$J$$ with respective to $$v$$.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/CG2.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Here, we want to see how a minor change in $$v$$ would affect the value of $$J$$. We know that $$J = 3v$$ which in this case is 33. Increasing it by an infinitesimally small value gives us

$$\frac{dJ}{dv} = 3$$

Now, suppose we want to see $$dJ/da$$,
we know $$a=5$$ which we bump up to $$a=5.001$$. This changes $$v=11$$ to $$v=11.001$$
This changes J from $$J=33$$ to $$J=33.003$$
Hence the increase in $$J$$ is 3 times the increase in $$a$$ hence giving us

$$\frac{dJ}{da}=3$$

Another way to see this is that a change in $$a$$ will result in a change in $$v$$ which will ultimately change $$J$$.

So, by changing $$a$$ we end up increasing $$v$$ by $$dv/da$$, whose change will cause $$J$$ to increase by $$dJ/da$$.

$$\frac{dJ}{da} = \frac{dJ}{dv}\frac{dv}{da} \;\;\text{(here we see that dv/da=1)}$$

In calculus, this is known as the **chain rule**.

When writing code with backpropagation, we usually have a final output variable we want to optimize ($$J$$ in this case). A lot of these computations involve finding the derivatives of the final output variable with respect to another intermediate variable.

In software we can name these with long variable names like `dJdvar`. But since we are always taking derivatives with respect to $$dJ$$, we just use `dvar` to represent that quantity.

So `da` would be $$dJ/da$$, `dv` would be $$dJ/dv$$ and so on.

Hence, performing the same for all the variables in our computation graph, would get us the following results.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/CG3.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Here, computing `dv` helps us compute `da` and `du`, and computing `du` helps us compute `db` and `dc`.

## Gradient Descent: Logistic Regression

Considering the case of Logistic Regression, we can explain the process of gradient descent in a relatively simple manner using computation graphs.

So, in the case of logistic regression, we have:
- The **linear part** given by $$z = w^{T}x + b$$
- The non-linear **activation**, giving us an output: $$\hat{y} = a = \sigma (z)$$
- The Loss: $$L(a,y) = -(y \log{a} + (1-y)\log{(1-a)})$$

where $$a$$ is the output of the logistic regression model and $$y$$ is the ground truth label.

### Gradient Descent on a Single Example

In this example taking 2 inputs $$x_{1},w_{1}$$ and $$x_{2},w_{2}$$ along with bias $$b$$. The computation graph for this is seen as

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/LR_GD.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Here, we modify the parameters $$w$$ and $$b$$ to reduce the loss $$L$$. This process is the **forward propagation** for a single training example.

For **backward propagation**, we want to compute the derivatives with respect to loss $$L$$. To do this, we first go backwards to $$a$$ computing $$dL/da$$ aka `da`. Computing this, we get

$$
\begin{align*}
& \frac{dL}{da} = \frac{d}{da}(-(y \log{a} + (1-y)\log{(1-a)}))\\
\Rightarrow & \boxed{\frac{dL}{da} = -\frac{y}{a} + \frac{1-y}{1-a}}
\end{align*}
$$

Now that we have `da`, we can compute `dz` as

$$
\begin{align*}
&\frac{dL}{dz} = \frac{dL}{da}\;\frac{da}{dz} \\
\\
&\frac{da}{dz} = \frac{d}{dz}\left(\frac{1}{1+e^{-z}}\right) \\
\Rightarrow&\frac{da}{dz} = \frac{e^{-z}}{(1+e^{-z})^{2}} \\
\Rightarrow&\frac{da}{dz} = \left(\frac{1}{1+e^{-z}}\right)\left(1-\frac{1}{1+e^{-z}}\right) \\
\Rightarrow&\boxed{\frac{da}{dz} = a(1-a)} \\

\\
&\frac{dL}{dz} = \frac{dL}{da}\;\frac{da}{dz}\\
\Rightarrow&\frac{dL}{dz} = \left(-\frac{y}{a} + \frac{1-y}{1-a}\right)(a(1-a))\\
\Rightarrow&\frac{dL}{dz} = -y(1-a) + a(1-y)\\
\Rightarrow&\boxed{\frac{dL}{dz} = a-y}\\
\Rightarrow&\boxed{\frac{dL}{dz} = \frac{dL}{da}*a*(1-a)}
\end{align*}
$$

The final step is to calculate how much $$w$$ and $$b$$ need to change. Computing `dw1`, `dw2` and `db` we get

$$
\begin{align*}
&\frac{dL}{dw_{1}} = \frac{dL}{dz}\;\frac{dz}{dw_{1}} \\
\\
&\frac{dz}{dw_{1}} = \frac{d}{dw_{1}}(w_{1}x_{1}+w_{2}x_{2}+b)\\
&\frac{dz}{dw_{1}} = x_{1}\\
\\
\Rightarrow&\boxed{\frac{dL}{dw_{1}} = x_{1}\frac{dL}{dz} = x_{1}(a-y)}\\
\\
&\text{Similarly}\\
\\
&\boxed{\frac{dL}{dw_{2}} = x_{2}\frac{dL}{dz} = x_{2}(a-y)}\\
\\
&\text{and}\\
\\
&\boxed{\frac{dL}{db} = \frac{dL}{dz} = (a - y)}
\end{align*}
$$

Here, we see that
- `dw1 = x1 dz`
- `dw2 = x2 dz`
- `db = dz`

If we want to perform the gradient descent for this example, we use the above formulae to compute `dz`, using which we would compute the resultant formulae for `dw1`, `dw2` and `db` after which we update $$w_{1}$$, $$w_{2}$$ and $$b$$ as

$$
\begin{align*}
w_{1} &:= w_{1} - \alpha \; dw_{1}\\
w_{2} &:= w_{2} - \alpha \; dw_{2}\\
b &:= b - \alpha \; db
\end{align*}
$$

### Gradient Descent on Multiple Examples

Considering $$m$$ number of examples as opposed to a single example to work with, the cost function for gradient descent is given by

$$
J(w,b) = \frac{1}{m}\sum\limits_{i=1}^{m}L(a^{(i)},y^{(i)})
$$

where $$a^{(i)} = \hat{y}^{(i)} = \sigma(z^{(i)})= \sigma(w^{T}x^{(i)} + b)$$.

The overall cost function is the average of the losses, so the derivative with respect to $$w_{1}$$ is also seen to be the average of derivatives with respect to $$w_{1}$$ of the individual loss terms.

$$
\frac{\partial}{\partial w_{1}}J(w,b) = \frac{1}{m}\sum\limits_{i=1}^{m} \underbrace{\frac{\partial}{\partial w_{1}} L(a^{(i)},y^{(i)})}_{dw^{(i)} - (x^{(i)},y^{(i)})}
$$

Algorithmically, This would be:
- Initialize `J=0`, `dw1=0`, `dw2=0` and `db=0`.
- `for i = 1 to m` over the training set
	- Compute derivative for each training example `z_i = w_T * x_i + b`
	- `a_i = sigma(z_i)`
	- `J += -((y_i * log(a_i)) + ((1 - y_i)* log(1 - a_i)))`
	- `dz_i = a_i - y_i`
	- `dw1 += x1_i * dz_i`
	- `dw2 += x2_i * dz_i` (do this for all features, in this case we only take 2)
	- `db += dz_i`
	- `J /= m`
	- `dw1 /= m`
	- `dw2 /= m`
	- `db /= m`
- Then we update the weights accordingly
	- `w1 := w1 - (alpha * dw1)`
	- `w2 := w2 - (alpha * dw2)`
	- `b := b - (alpha * db)`
- Update `J` with the new values of `w1`, `w2` and `b`.
- Repeat this for multiple steps of gradient descent

Here `J`, `dw1`, `dw2` and `db` act as accumulators to sum over the entire training set.

Two _weaknesses_ with the calculations above are.
- If we implement Logistic Regression this way, we use two for loops. Using explicit for loops in the code makes the algorithm less efficient.
- With large datasets, we need to use techniques such as **vectorization** to improve the performance.

# Single Layer Neural Networks

Now that we have an idea about the basic steps of training neural networks, we can now look at single layer neural networks to understand how they work.

The most common type of artificial neurons used in today's day and age is the **sigmoid neuron** which we have discussed [earlier](#logistic-regression).

We can picture the sigmoid neuron as follows

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/SNN1.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A single neuron.
</div>

Here we have the inputs $$[x_1,x_2,x_3]$$ which are stacked vertically. This is known as the **input layer**. These values are input to a node which computes the linear $$z$$ value, and the non-linear activation $$a$$ to give us the output prediction $$\hat{y}$$ and loss $$L$$.

A neural network can be seen as a stacking of several such functional units as shown below.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/SNN2.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A shallow neural network.
</div>

This is known as a **shallow neural network**.

Here, the first stack on nodes would correspond to a $$z$$-like calculation as well as an $$a$$-like calculation, the next node will correspond to another $$z$$ and $$a$$ like calculation, giving us a prediction $$\hat{y}$$.

By varying the weights and thresholds we get different decision making models.

The layer of circles shown in the above image after the input is known as the **hidden layer**. The name hidden refers to the fact that the true value of the weights and biases of these nodes are not observed. We mainly focus on the inputs and outputs when working with neural networks.

The final layer here is a single node, called the output layer, which is responsible for generating the predicted value $$\hat{y}$$.

## Neural Network Notations

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/SNN3.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

- Each layer of the neural network are represented with superscript square brackets $$[1],[2],\dots$$. The input features of the network are referred to as $$A^{[0]}$$ which are passed to the first hidden layer
- The first hidden layer generates a set of linear functions given by $$z^{[1]}$$ consisting of weights and biases $$W^{[1]}$$ and $$b^{[1]}$$, which are fed to an activation function giving us a set of outputs $$a^{[1]}_i$$ where $$i$$ refers to each node. For any subsequent hidden layers we would represent these values with the respective layer numbers in the superscript square brackets.
- The output layer in this example has only one node which generates the output $$a^{[2]}$$, which in this case can be seen as $$\hat{y}=a^{[2]}$$.

The network in this case is called a **two layer neural network** or a **single hidden layer neural network** because when counting the layers in a neural network, we do not count the input layer. The hidden layer here is layer 1 and the output layer here is layer 2.

Each of the two here layers will have parameters associated with them such as $$w^{[1]}, b^{[1]}$$ for the hidden layer. $$w$$ will be a $$(4,3)$$ matrix in this case and $$b$$ will be a $$(4,1)$$ vector. Similarly the output layer will have $$w^{[2]}, b^{[2]}$$ of dimensions $$(1,4)$$ and $$(1,1)$$.

# Computing the Output of a Neural Network

The process of computing the output of a neural network is similar to logistic regression but repeated multiple times at each node of the neural network.

We can visualize the calculations that take place in a node as given below

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/LogRegRepresentation.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Consider the first layer of a network. Here the computation happens in two steps
- Computing $$z^{[1]} = w^{[1]T}x + b^{[1]}$$
- Computing $$a^{[1]} = \sigma(z^{[1]})$$

Here $$z$$ is the linear part, $$a$$ is the non-linear activation function output, $$w$$ is the weight and $$b$$ is the bias.

For multiple nodes in a given layer, we would get the following calculations

$$
\begin{align*}
z_{1}^{[1]} &= w_{1}^{[1]T}x + b_{1}^{[1]} & a_{1}^{[1]} = \sigma(z_{1}^{[1]})\\
z_{2}^{[1]} &= w_{2}^{[1]T}x + b_{2}^{[1]} & a_{2}^{[1]} = \sigma(z_{2}^{[1]})\\
z_{3}^{[1]} &= w_{3}^{[1]T}x + b_{3}^{[1]} & a_{3}^{[1]} = \sigma(z_{3}^{[1]})\\
z_{4}^{[1]} &= w_{4}^{[1]T}x + b_{4}^{[1]} & a_{4}^{[1]} = \sigma(z_{4}^{[1]})\\
\end{align*}
$$

In practice, these equations are vectorized to compute $$Z$$ as a vector. If we had 4 logistic regression units stacked together into a layer, the calculation would look something like

$$
Z^{[1]} = \begin{bmatrix}
z_{1}^{[1]} \\
z_{2}^{[1]} \\
z_{3}^{[1]} \\
z_{4}^{[1]}
\end{bmatrix} =
\begin{bmatrix}
\dots & w_{1}^{[1]T} \dots \\
\dots & w_{2}^{[1]T} \dots \\
\dots & w_{3}^{[1]T} \dots \\
\dots & w_{4}^{[1]T} \dots
\end{bmatrix}
\cdot
\begin{bmatrix}
x_{1} \\
x_{2} \\
x_{3}
\end{bmatrix} +
\begin{bmatrix}
b_{1} \\
b_{2} \\
b_{3} \\
b_{4}
\end{bmatrix}
$$

This gives us the outputs for the first layer $$a^{[1]}$$.

# Activation Functions

**Activation functions** serve the purpose of introducing a non-linearity to our model, enabling it to learn complex patterns from the data given to it.

In the calculations being done in a neural network the activation functions look something like

$$
\begin{align*}
z^{[1]} &= W^{[1]}x + b^{[1]}\\
a^{[1]} &= g^{[1]}(z^{[1]})\\
z^{[2]} &= W^{[2]}a^{[1]} + b^{[2]}\\
a^{[2]} &= g^{[2]}(z^{[2]})\\
\end{align*}
$$

where $$g$$ is the activation function.

We will now look at some of the commonly used activation functions.

## Sigmoid Function

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Sigmoid.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

- The sigmoid function maps all the values input into it to a binary output.
- Equation: $$a = \frac{1}{1+e^{-z}}$$
- Value Range: $$[0,1]$$
- Issues: Sigmoid functions can result in vanishing gradient problems

## Hyperbolic Tangent (Tanh) Function

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Tanh.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

- Equation: $$a = tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$
- Value range: $$[-1,1]$$
- Pros: This function is works better than the sigmoid function as the activations have a zero mean, making the learning process easier.
- Con: This function also suffers from the vanishing gradient problem that the sigmoid function faces.
- This can be seen as a mathematically shifted version of the Sigmoid function.

## Rectified Linear Unit (ReLU) Function

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/ReLU.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

- Equation: $$a = max(0,z)$$
- Value range: $$[0,\infty]$$
- Pros: This function is fast to compute and mitigates the vanishing gradient problem.
- Con: For negative $$z$$ values the gradient is zero.
- The **Leaky ReLU** function allows for a small non-zero gradient when the value of $$z$$ is less than 0, improving the performance of the ReLU function.
- Leaky ReLU Equation: $$a = max(cz,z)$$
- Leaky ReLU Value Range: $$[-\infty,\infty]$$
- The modification here can be done by changing the slope for $$z < 0$$.

## General Rules

- Sigmoid functions are used for binary classification problems and are useful in the output layer
- ReLU is often the default function used for hidden layers. Tanh can sometimes be used too.
- Variants like leaky ReLU are preferred for improvements in performance of the network.

Data science problems are diverse hence needing for a lot of experimentation in terms of using different activation functions.

Without activation functions the neural networks' behavior would be similar to a linear regression model, being unable to capture the complexities of the data.

# Derivatives of Activation Functions

When performing backpropagation through a neural network, we need to perform derivatives on the outputs of the different layers. For this reason, it is useful to know how the derivatives of different activation functions are found.

## Sigmoid Function

Given the sigmoid activation $$g(z) = \frac{1}{1+e^{-z}}$$, we get $$\frac{d}{dz}g(z)$$ as

$$
g'(z) = \frac{1}{1+e^{-z}}\left(1 - \frac{1}{1+e^{-z}}\right)
$$

$$
\Rightarrow\boxed{\frac{d}{dz}g(z) = g(z) (1 - g(z))}
$$

The complete derivation for this has been discussed previously and can be found [here](#gradient-descent-on-a-single-example).

## Tanh Function

Given the tanh function $$g(z) = tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$, We find $$g'(z)$$ as

$$
\begin{align*}
&g'(z) = \frac{(e^{z} + e^{-z})^{2} - (e^{z} - e^{-z})^{2}}{(e^{z} + e^{-z})^{2}}\\
&g'(z) = 1 - \frac{(e^{z} - e^{-z})^{2}}{(e^{z} + e^{-z})^{2}}\\
\Rightarrow &\boxed{g'(z) = 1 - g(z)^{2}}
\end{align*}
$$

## ReLU Function

Given the ReLU function $$g(z) = max(0,z)$$, we can find the derivative of it to be

$$
g'(z) = \begin{cases}
0 & \text{if } z < 0 \\
1 & \text{if } z \geq 0 \\
\text{undefined} & \text{if } z = 0
\end{cases}
$$

In software, the last part won’t necessarily be correct and it works just fine at $$z = 0$$.

## Leaky ReLU

Here, the activation function is $$g(z) = max(cz,z)$$
hence giving us:

$$
g'(z) = \begin{cases}
c & \text{if } z < 0, c \in (0,1) \\
1 & \text{if } z \geq 0
\end{cases}
$$

# Weight Initialization in Neural Networks

When building a Neural Network it is important to ensure that the initial weights and biases selected are done so randomly.

Consider a case with two input features and two hidden units. The input matrix $$w^{[1]}$$ will be of $$(2,2)$$ dimension.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Random_Init_Example.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Here, we take

$$
w^{[1]} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}
$$

$$
b^{[1]} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

Here, for any example

$$
a^{[1]}_{1} = a^{[1]}_{2} = 0
$$

And, when computing backpropagation we get

$$
dz^{[1]}_{1} = dz^{[1]}_{2} = 0
$$

We see that initializing the bias to zero is fine, but initializing the weights to zero causes issues.

Here, the hidden units will be completely identical as they turn out to be the same function. Hence, after training, our neural network will have the two hidden units, computing the exact same function, having the same amount of influence on their outputs. Hence, the updates performed will also be exactly the same.

In addition, zero weights make it likely that the neurons are activated in flat regions of the activation function, hence leading to a small gradient value, leading to the vanishing gradient problem.

## How Random Weights Help

- Random initialization of the weights ensures that each neuron has a unique starting linear function, hence computing different activations, breaking the symmetry. This allows the network to learn from the errors, making updates on each neuron in the network.
- Using small initialization values ensures that the activation functions operate in ranges where the gradients aren't flat.

## Weight Initialization in Practice

- In practice the weights are initialized to small random values multiplied by a constant like $$0.01$$ to ensure the values are small. In the case of sigmoid and tanh functions, this ensures that the values taken up by the weights are within the working regions for these functions
- For biases, it is fine to initialize the values to zero as the weights themselves being initialized randomly is enough to break the symmetry.
- For deeper networks there exist advanced initialization techniques, but small random initialization values are useful with shallow networks.

# Deep Neural Networks

**Deep neural networks (DNNs)**, as opposed to shallow networks are capable of learning complex functions. However, it is hard to predict how many layers deep our network would have to be. Hence, we normally start with a single layer and move on from there, gradually increasing the number of layers.

The number of layers can be viewed as a hyperparameter which we use to evaluate our model.

## Notations for DNNs

Consider a 4 layer NN as shown below, with 3 hidden layers.

<div class="row justify-content-center mt-3">
    <div class="col-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/DNN1.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The notation here would be:
- $$L$$ : to denote the number of layers in the network (4 in this case)
- $$n_{x}$$ : gives us the number of input features
- $$n^{[l]}$$ : gives us the number of nodes/units in a layer $$l$$. Here $$n^{[1]} = n^{[2]} = 5, n^{[0]} = n_{x} = 3$$ and so on
- $$a^{[l]}$$: activations in layer $$l$$
- $$a^{[l]} = g^{[l]}(z^{[l]})$$: Activation function indexed by $$l$$
- $$w^{[l]}$$: weights for $$z^{[l]}$$ in the given layer $$l$$
- $$b^{[l]}$$: biases for the layer $$l$$
- $$x = a^{[0]}$$: inputs
- $$\hat{y} = a^{[L]}$$: output predictions

## Forward Propagation

We have already seen how [forward propagation](#forward-propagation) works in shallow neural networks. This process passes the input data to the network, computing the activations for each layer sequentially from the input layer to the output layer which gives us a prediction, classification or any other result that a network can produce.

The forward propagation steps for a deep neural network are as follows:
- **Input layer**: The input layer's data is fed into the network serving as the initial activations for the first layer of the network
- **Linear Combination**: For each neuron in the hidden layers, a weighted sum of the inputs are calculated with the associated weights and biases associated with them. For a layer $$l$$, the linear combination formula is $$z^{[l]} = w^{[l]}a^{[l-1]} + b^{[l]}$$, where $$w^{[l]},b^{[l]}$$  are the weights and biases of the current layer and $$a^{[l-1]}$$ is the activation from the previous layer.
- **Activation Function**: The linear part obtained here is input to the activation function which introduces the non-linearity in the system, helping it learn complex patterns from the data.

For a single training example, this process would look like:

$$
\begin{align*}
z^{[l]} &= w^{[l]}a^{[l-1]} + b^{[l]}\\
a^{[l]} &= g^{[l]}(z^{[l]})
\end{align*}
$$

When implementing this in code, the general rule of thumb is to avoid the use of for loops and to use vectorization for matrix multiplication on multiple examples. However, in this case for propagating over multiple layers from $$1$$ to $$L$$, this is unavoidable.

## The Building Blocks of DNNs

We have already seen how forward propagation and backward propagation are the key components needed for a neural network to work, and how putting these together allow for it to train on the data provided to it.

Considering a single layer $$l$$ of a DNN, the process of training it with weights and biases $$W^{[l]}, b^{[l]}$$ are as follows:

#### For forward propagation

- Input the previous activation $$a^{[l-1]}$$
- Calculate linear part and activation
    - $$Z^{[l]} = W^{[l]}a^{[l]}+b^{[l]}$$
    - $$a^{[l]} = g^{[l]}(Z^{[l]})$$
- Cache $$Z^{[l]}$$ for later use

#### For backward propagation

- Input $$da^{[l]}$$
    - $$da^{[l-1]} = W^{[l]T}dZ^{[l]}$$
- Output $$da^{[l-1]},dW^{[l]}$$, and $$db^{[l]}$$
    - $$dZ^{[l]} = da^{[l]} \times g^{[l]'}(Z^{[l]}) \Rightarrow dZ^{[l]} = W^{[l]T}dZ^{[l]} \times g^{[l]'}(Z^{[l]})$$
    - $$dW^{[l]} = dZ^{[l]}a^{[l-1]T}$$
    - $$db^{[l]} = dZ^{[l]}$$
- for this we use the cached $$Z^{[l]}$$ value and in addition to outputting $$da^{[l-1]}$$, we take the gradients we want in order to implement gradient descent for learning.

This process can be visualized as follows

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/DNN4.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Each of the units visualized here can be chained together to give us the whole process of the forward and backward propagation processes.

Conceptually, it is useful to think of the cache as storing the $$Z$$ values for the backward functions to use. When implementing this, however, we see that the cache is a convenient way to get the $$W^{[1]},b^{[1]}$$ into the backward function as well. Hence, we store the $$W$$ and $$b$$ values alongside $$Z$$ in the cache.

# Parameters and Hyperparameters

Deep learning involves the careful tuning of the parameters (weights and biases) and hyperparameters.

Parameters are intrinsic parts of the model and are learned and optimized from the input data during training.

Hyperparameters are set manually and help us govern the process of learning the parameters.

A rundown of some of the parameters and hyperparameters we work with when building a DNN are as follows:
1. Parameters
- Weights, $$W$$ and biases, $$b$$ are the main parameters in a neural network that get updated during the training process to minimize the error.
- Coefficients in the simple linear regression model($$y=mx+c$$) are another example of parameters where the slope ($$m$$) and y-intercept ($$c$$) are the parameters.
- In support vector machines, the support vectors, which are data points closest to the decision boundary, are the parameters.
2. Hyperparameters
- The **Learning Rate ($$\alpha$$)**: This determines he size of the steps taken during the optimization process.
- The **number of iterations** the optimization algorithm runs for.
- The **number of hidden layers (L)** which add to the depth and overall complexity of the network.
- The **number of hidden units** in each layer, which determines the size of each layer.
- The **activation functions** used which add non-linearity to the model, hence allowing it to learn from the losses generated at each layer.

Hyperparameters affect the efficiency and accuracy of the model when training the parameters $$W$$ and $$b$$. Due to this, it is important that they are selected appropriately.

# Why Neural Networks Work Well

Consider an example where we are building a face detection algorithm with a large number of examples to work with. The layers of the network would work something like the following:
- The **first layer** would work on detecting edges in several orientations like vertical, horizontal, at different angles, and so on.
- The **second layer** can group together features from the first layer to form slightly complex facial features like eyes, lips, ears and so on.
- The **third and later layers** can then take up the task of combining facial the features from the previous layer to generate examples of faces that can be used to detect faces in the output layer.

Similarly for speech recognition
- The **first layer** would work on detecting low level audio features in the form of waveforms.
- The **second layer** can group together the waveforms from the first layer to compose the low level units of sound (phenomes).
- The **third and later layers** can then work on generating words, phrases and sentences based on the inputs fed to it.

This process can be described as **hierarchical feature learning**, where we see that the problem being worked on is broken down into smaller units from which more and more complex features are being built. This resembles how we believe a human brain processes information, however, it is not necessarily an accurate comparison as the way the brain works is still unknown to medical professionals to this day.

Deep learning and Artificial Intelligence have garnered a lot of public interest as of late, being a branding catchphrase used by many companies and organizations for their products. This is rightfully so as deep learning, though still being a bit rough around the edges have demonstrated superior performance in several fields of application.

While we know that deep learning models work well, we need to first understand whether a given problem warrants the use of such an architecture. It is always a good practice to first use simpler models like decision trees, logistic regression or networks with a few layers before moving on to more complex and deep architectures.

# Conclusion

In this article, we have seen how deep artificial neural networks work from the ground up, gaining an understanding of the underlying mathematics and how they have the potential to learn increasingly complex functions, hence finding a great deal of use in several fields.

Now that we have these concepts down, it would be good to know how the training process for more complex neural networks would work. The next post will highlight this topic. We will also look at an implementation of a neural network from scratch and comparing its performance with pre-made deep learning frameworks such as Tensorflow and PyTorch.