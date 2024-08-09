---
layout: post
title: Deep Learning - 1
date: 2024-06-30 16:40:16
description: "Topics covered: Introduction to Neural Networks and Deep Learning"
tags: machine-learning deep-learning python
categories: deep-learning
toc:
  sidebar: left
bibliography: 2024-06-30-dl-1.bib
related_publications: true
---

This is the first of a series of posts that will compile my reports for a Deep Learning independent study taken up at the University of Massachusetts Dartmouth.

This study is loosely based on the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) offered by [deeplearning.ai](https://www.deeplearning.ai/) with more content included from outside sources where necessary. The focus here is to start from the foundations to help understand the capabilities, challenges and consequences of deep learning, gradually moving on to more complex topics and applications of these concepts. The main goal is to gain insights on the inner workings of the concepts behind cutting-edge AI models used in various fields as well as hands-on experience using these models. Equal amounts of importance will be give to the underlying logic and mathematics behind various deep learning models and their practical applications.

This post focuses on the basics of deep learning and neural networks with a focus on the underlying computation. Alongside this, we will also look at a project implementing these concepts from scratch on classifying image data.

<hr>

# Introduction to Deep Learning

In the world of data science{% cite coursera2024 %}, we usually see mentions of statistics, machine learning (ML), artificial intelligence (AI), and deep learning several times. Though machine learning and deep learning are both subfields of AI and both terms are used interchangeably, they have their own meanings.

**Machine learning** algorithms are relatively simple models designed to adapt on input data and either find patterns or make decisions on newly provided data.

**Deep learning** on the other hand, is a branch of machine learning that involves training expansive *artificial neural networks (ANNs)*. An ANN consists of layers of interconnected nodes called *neurons* that, together, help process and learn patterns from the input.

We often picture AI, deep learning and machine learning as a venn diagram with AI taking up the outermost layer, machine learning being the intermediate layer, and the innermost layer being deep learning. A more detailed visualization of this can be seen below. 

Deep learning has not only transformed how we search for content on the internet, but has also found its use in domains such as:
- **Healthcare**:{% cite DLNIH %} Deep learning is used in several fields such as biomedical informatics, genomics and public health to gain insights from high dimensional and complex data. Several challenges and opportunities are faced in this field, especially when developing data-driven tools for knowledge discovery and predictive analysis. Deep Learning finds its use here as it is capable of creating models that can more accurately represent the complex relationships in the data and create meaningful abstractions from it.
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
        {% include figure.liquid loading="eager" path="assets/img/housing_example_2.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The modified linear regression line with truncation (a.k.a ReLU).
</div>

This resulting function is known as a **Rectified Linear Unit** or **ReLU** for short, and is widely used in the development of deep learning models.

##### Scenario 2

Now, suppose we have more features to work with in addition to the size of the house, such as the number of *bedrooms*, *zip code* and *wealth* of the neighborhood. In this scenario we can make the following observations:
- The *size* and *number of bedrooms* gives us an idea of whether a house can accommodate for the *family size*.
- The *zip code* tells us how *walkable* the locality of the house is.
- The *zip code* along with the *wealth* of the neighborhood can give us an idea of the *quaity of schools* in the area.

All of this information can be visualized in the form of a neural network as shown.

<div class="row justify-content-center mt-3">
    <div class="col-10 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/housing_example_nn_1.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The circles in the image here represent **nodes** which can contain a ReLU or any other function. The arrows here represent the flow of information between the input and nodes.

All the observations made above can be seen as functions that take in the information to give the corresponding outputs for family size, walkability and school quality. In a similar fashion, all of these can together be taken in as inputs for yet another node which gives us an estimate for the price of the given house.

This example gives us a look into how people can go about deciding how much they would be willing to spend on a house.

##### Scenario 3 - A real-world Neural Network

The inputs here can be considered as a vector $$X$$ with the target (price) taken as $$y$$ which we aim to predict.

In practice, we give the neural network the $$X$$ values along with the target $$y$$ as the training set. As opposed to the previous scenario, the function of the intermediate layers are figured out by the neural network by itself.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/actual_nn.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A real-world implementation of a single layer neural network.
</div>

A practical application of a neural network would look like the image above. The circles here are called **hidden units** which are arranged in layers known as **hidden layers**.

Each node here takes inputs from all of the input features. We do not have control over what each node does. The neural network itself makes the decision on what each node is supposed to do.

In addition, a practical neural network will seldom have just one neuron or a single layer. Instead a collection of several neurons receiving differing inputs would be used, which collaborate to give an accurate and complex prediction.

In this case, the layer  is known as a **densely connected layer** as every input feature is connected to every node in in the layer.

The main advantage of neural networks is that with enough data about the input features and the target, and enough training examples for them, neural networks are effective at figuring out functions to map the inputs to the outputs. 

# Supervised Learning with Neural Networks



# Logistic Regression

## Gradient Descent

## Derivatives (Not necessary)

# Single Layer Neural Networks

## Neural Network Notations

# Computing the Output of a Neural Network

# Activation Functions

# Derivatives of Activation Functions

# Implementing Gradient Descent in Neural Networks

# Backpropagation in a Two-layer Neural Network

# Summary

<!--####

Pug heirloom High Life vinyl swag, single-origin coffee four dollar toast taxidermy reprehenderit fap distillery master cleanse locavore. Est anim sapiente leggings Brooklyn ea. Thundercats locavore excepteur veniam eiusmod. Raw denim Truffaut Schlitz, migas sapiente Portland VHS twee Bushwick Marfa typewriter retro id keytar.

> We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another, unevenly. We grow partially. We are relative. We are mature in one realm, childish in another.
> —Anais Nin

Fap aliqua qui, scenester pug Echo Park polaroid irony shabby chic ex cardigan church-key Odd Future accusamus. Blog stumptown sartorial squid, gastropub duis aesthetic Truffaut vero. Pinterest tilde twee, odio mumblecore jean shorts lumbersexual. -->
