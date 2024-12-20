---
layout: post
title: Deep Learning - Improving Deep Neural Networks
date: 2024-10-22 10:58:00
description: "Topics covered: Deep Learning in practice, Optimization Algorithms, Hyperparameter Tuning, and Batch Normalization."
tags: machine-learning deep-learning
categories: deep-learning
toc:
  sidebar: left
bibliography: 2024-06-30-dl-1.bib
related_publications: true
pretty_table: true
---

This blog post is the second of a series of posts covering deep learning from the fundamentals up to more complex applications.

Here we will look at the practical aspects of deep learning when training deep networks, covering topics such as regularization, optimization algorithms, hyperparameter tuning, batch normalization and more, which will equip us with  knowledge needed when implementing deep neural networks in the real world.

<hr>

# Practical Aspects of Deep Learning

The performance of our machine learning and neural network models depend a lot on how our dataset is set up.

In practice the development of machine and deep learning models is an **iterative process** starting with an **idea**, which is then implemented in **code** using which we **experiment** on our idea to see how our implementation works. We can than refine the idea we have from the outputs of our experiment, leading to a positive feedback loop with the goal of improving our neural network.

Some of the factors we work with in deep learning include the depth of the network, number of hidden units per layer, the learning rates, regularization terms, activation functions and so on.

Given that deep learning has found success in many different domains, it has been found that the knowledge transferability between different domains is limited. For example, someone working in natural language processing would find that the intuitions from that field would not work the best in computer vision. This could be due to the difference in the parameter values being used used as well as the model architectures being used in the fields, with sequential models being used for NLP and spatial hierarchies being used in computer vision.

## Splitting Up Our Data

The traditional machine learning approach to split data is to split it such that one part acts as the **training set**, one as the **hold-out cross validation** set (also known as the **development set**) and the **test set**.

The training set, as the name suggests, is used to train our algorithms, the dev set helps us see which model performs the best and improve it accordingly, and the test set is used to get the final unbiased estimate on how well the algorithm is doing.

The common split used to be a **70/30** train/test split or a **60/20/20** train/dev/test split.

However, with the advent of Big Data, with millions of data points to work with, using a much smaller percentage of the total dataset for the dev and test sets is preferred. This split would be more in the vein of a **98/1/1** split or even lower, like **99.5/0.25/0.25** for much larger datasets.

This is because the dev sets' use is only to check which model works well and to make a choice. Using a large amount of the data for this would lead to us missing out on information that could be used to train our model.

Similarly the test set's job is only to tell us how our final model is working.

## Ensuring Consistency in Dev/Test Distributions

The data we train our model on can come from various sources such as images from the web and user submitted images for a computer vision application. These images can have vast differences in terms of their content and quality, hence leading to a huge difference in the distributions of the data.

The general rule of thumb is to ensure that the dev and test sets are are from the same distributions to ensure that the models' performance metrics obtained on the dev set are representative of the its performance on the test set.

## The Test Set

The test set gives us an unbiased view into how our model performs. However, in some cases, it is alright not use it altogether.

However, not using a test set leads to the risk of reduced performance of the model due to overfitting. Hence, it is better to use the test set whenever necessary and avoid using it excessively when tuning our parameters.

It is important that the model we are training is iterated upon rapidly. The faster a model can be trained, the quicker we can reach the optimal performance.

This can be done so through effectively setting up train/dev/test sets.

## Bias/ Variance

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dataset.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Examples of bias and variance effects on the mode (This image was drawn referencing Coursera)
</div>

**Bias**: A high bias error occurs due to an overly simple model fitting the data, such as a simple linear or logistic regression curve, leading to it not capturing the true patterns in the data, hence performing poorly. This can also be seen as **underfitting**.

**Variance**: A high variance error occurs when the model developed has a high level of complexity, leading to a large amount of noise being included in the models' predictions. This leads to a model highly tailored to the training data, leading to poor performance on the testing data. This is also known as **overfitting**.

With the advent of deep learning and the ability to build deep networks, it has become possible to reduce both the bias and variance. This leads us away from the classical machine learning notion of there being an unavoidable trade-off between the bias and variance.

### Diagnosing bias and variance issues using errors

A low **training set error** can be an indicator that the model fits the training data well but, by itself it doesn't tell us whether our model is overfitting the data.

This is where the **development set** comes in. This set helps is understand whether our model does a good job at generalizing over unknown data.

A few scenarios we can look at regarding the training and dev set errors are:
- A **low training and high dev error** is indicative of our model _overfitting_ the data, having a _high variance_.
- A **high training and dev error** indicates that our model is _underfitting_ the data suggesting a _high bias_.
- If **both errors are high and close** then the model may have _both a high bias and variance_.
- If **both errors are low**, this is the ideal outcome which tells us that our model is _performing well_.

The analyses of these errors are based on the assumption that the human error or the lower limit of possible error for the model, a.k.a **Bayes Error** is $$0$$.

If the Bayes error is higher then we can say that some of our created models are reasonable to use.

The main takeaway here is that recognizing and diagnosing whether our model suffers from a high bias or variance is crucial to ensuring our model fits the training data and generalizes well to new data. We can easily diagnose these from the training and dev errors alongside the Bayes and human errors.

### Addressing bias and variance issues in neural networks


A **high bias** suggests that our model is too simple and is underfitting the data. The solutions for this could be:
- **Using a bigger network**: This can be achieved by adding more layers or hidden units to our network. This makes it possible to capture more complex patterns in our data but also makes our model more computationally expensive to train.
- **Prolonged training**: Training our model for a larger number of iterations gives it more time to capture additional information from the data, hence  leading to a better fit. However, one must be careful as a model left to train for too long could potentially lead to overfitting.
- **Advanced Optimization Algorithms**: These are methods that can speed up the process of training our model, helping it converge and fit the data better. [Click here](#optimization-algorithms) to learn more about them.
- **Trying other network architectures**: Some problems may call for the use of different neural network structures. It is always a good practice to experiment with different architectures to see which ones fit.

A **high variance** tells us that our model is too complex and is capturing noise instead of the underlying pattern in the data, hence leading to overfitting. Some solutions for this could be:
- **Procuring more data**: Obtaining more data for our model to work on is one of the most effective ways to combat a high variance. This is because it gives the model more examples to generalize from.
- **Regularization**: Regularization penalizes our model when it becomes overly complex, hence reducing overfitting.
- **Network Architecture Tweaking**: Similar to solving the bias problem, tweaking the network architecture is also useful in reducing the variance.

### Bias-Variance Trade-off

Classical machine learning and statistics sticks to the belief that the bias and variance are intrinsically linked, and a decrease in bias would increase the variance and vice versa. This is known as the **bias-variance trade-off**.

In the era of deep learning, a large network is enough to reduce the bias when working with larger volumes of data without affecting the variance as, long as regularization is implemented properly.

# Regularization

**Regularization** is one of the fundamental techniques used in machine learning to reduce the complexity of the model being developed, hence preventing it from overfitting the data.

Obtaining more training data is normally what one would do if they wanted to combat the variance, however, obtaining more training data in the real-world is not always feasible. This is where regularization finds its role.

For this section we will use logistic regression as an example. We have already seen in the [previous blog]({{site.baseurl}}{% link _posts/2024-09-07-deep-learning-self-study-1.md %}), that the main goal of logistic regression is to minimize the cost function $$J(w,b)$$ based on the model's decisions

$$
J(w,b) = \frac{1}{m}\sum\limits_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)})
$$

where $$w \in \mathbb{R}^{n_{x}}$$ and $$b \in \mathbb{R}$$.

We will now look at how regularization can help deal with the variance issue in this example.

## L2 Regularization

L2 regularization adds an additional term to the logistic regression cost function as follows:

$$
J(w,b) = \frac{1}{m}\sum\limits_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)}) + \boxed{\frac{\lambda}{2m}\|w\|_{2}^{2}}
$$

where

$$
\boxed{\|w\|_{2}^{2} = \sum\limits_{j=1}^{n_{x}}w_{j}^{2} = w^{T}w}
$$

In this additional term, $$\lambda$$ is known as the regularization parameter and $$\|w\|_{2}^{2}$$ is the square of the Euclidean norm of the weight vector $$w$$.

We only normalize $$w$$ as it is commonly a high dimensional vector with many more parameters while the bias $$b$$ is just a single number. Hence changing the weight vector has more of a significant effect on the overall model than the bias.

L2 regularization is the most commonly used type of regularization and is also known as **weight decay**.

## L1 Regularization

In the case of L1 regularization, the cost function is modified ass follows

$$
J(w,b) = \frac{1}{m}\sum\limits_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)}) + \boxed{\frac{\lambda}{m}\sum\limits_{i=1}^{n_{x}}\|w\|_{1}}
$$

where

$$
\boxed{|w||_{1} = \sum\limits_{i=1}^{n_{x}} \lvert w \rvert}
$$

L1 regularization adds the absolute value of the weight vectors to the weight parameters as opposed to L2 regularization. This leads to sparse weight vectors as many of the parameters are set to zero.

As a result, the overall model produced is compressed due to many of the nodes effectively being removed.

L1 regularization is not as effective in comparison to L2 regularization due to which it is not used as much in comparison.

#### The role of $$\lambda$$

The **regularization parameter** helps balance the trade-off we make between fitting the training data and preventing overfitting by keeping the weights small.

## Regularization in Neural Networks

In a neural network, the cost function is a function that encompasses all the parameters across all the layers in our model. It is given by

$$J(w^{[1]},b^{[1]},\dots,w^{[L]},b^{[L]}) = \frac{1}{m}\sum_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}\sum\limits_{l=1}^{L}\|w^{l}\|^{2}$$

Where

$$
\|w^{[l]}\|_{F} = \sum_{i=1}^{n^{[l]}}\sum\limits_{j=1}^{n^{[l-1]}}(w^{[l]}_{ij})^{2}
$$

The squared norm here, which is the sum of squares of all the entries, is known as the **Frobenius norm**.

#### Gradient descent with regularization

Similar to how we would normally go about it, we calculate `dw` using the backpropagation, taking the partial derivatives of the cost function with respect to $$w$$ for any given $$l$$ layer. The only difference is that in this case, we add the regularization term to the resultant term.

$$dW^{[l]} =  \frac{\partial{J}}{\partial{W^{[l]}}} + \frac{\lambda}{m}W^{[l]}$$

using which we update the parameters as follows

$$W^{[l]} := W^{[l]} - \alpha \left[ dW^{[l]} + \frac{\lambda}{m}W^{l} \right]$$

$$\Rightarrow W^{[l]} := W^{[l]}  \left[ 1 - \alpha\frac{\lambda}{m}W^{l} \right] - \alpha dW^{[l]}$$

Here, we **effectively reduce the values of the matrix** a bit as we are reducing the $$W^{[l]}$$ values by a factor of $$[1 - \alpha\frac{\lambda}{m}W^{l}]$$. This justifies the name "weight decay" for L2 normalization.

## How Regularization works

Consider a large neural network which is overfitting with the cost function

$$J(W^{[l]},b^{[l]}) = \frac{1}{m}\sum\limits_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)})$$

When using the regularization term, the cost function becomes

$$J(w,b) = \frac{1}{m}\sum\limits_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}\sum\limits_{l=1}^{L}||w^{l}||^{2}$$

The way regularization works is, if we crank our regularization parameter $$\lambda$$ to be really big, we are incentivized to set the weight matrices $$W$$ close to 0.

The idea is that if the weights are set extremely low, a lot of the hidden units will effectively be zeroed out, hence leading to them having not much of an impact on the output received.

This effectively gives us a much smaller neural network, hence taking us from overfitting the data more towards increasing the bias.

## Dropout Regularization

In addition to L2 regularization, another regularization method used is called dropout regularization.

### Intuition behind dropout

Consider a neural network that is overfitting its data.

During the forward pass, dropout goes through each of the layers, setting a probability to each of the nodes to temporarily remove/disable it, or "drop it out". After this, backpropagation is performed on the diminished network. This leads to training being done on a slightly different network on every pass.

Looking at it from the point of view from a single unit taking in $$4$$ inputs to generate a meaningful output, dropout randomly eliminates some of the inputs at every iteration.

Due to this, with every iteration some of the input features can be present or absent at random. This  leads to the node not relying on any one feature as it may or may not be present in subsequent iterations of the training process.

This leads to an overall shrinkage in the squared norm of the weights.

Dropout can be seen as an adaptive form of L2 regularization but the L2 penalty on different weights are different depending on the size of activations being multiplied into the weight.

Dropout is done with the help of a parameter called the **keep probability**, based on which a coin toss is done on every node during the training and used to decide whether a given node will be kept or dropped from the network during a given iteration of training.

### Downsides of Dropout

- Despite every iteration of dropout learning being faster, we will need to train the network for a larger number of epochs to converge. This is because on every epoch, the training algorithm works on a subset of the whole network.
- Dropout only works well for larger networks and big datasets where overfitting is a huge possibility. For smaller networks and datasets dropout is not of much benefit and can harm the performance.
- Dropout adds an extra tuning parameter to work with, that is, the keep probability, which is used as the probability for dropping a node in the network.

## Other Regularization Methods

### Data Augmentation

**Data Augmentation** is the process of generating new data from pre-existing data to train our deep learning models. This is usually implemented when the data at hand is extremely limited in quantity and procuring new data is not feasible.

This adds a bit of redundancy to our dataset as it is not as effective as collecting new data, but it is cost efficient.

Some examples of data augmentation are:
- Flipping images horizontally
- Randomly cropping the image and distorting it
- Adding random rotations and distortions to images in the case of OCR.

### Early Stopping

When training a machine learning algorithm or neural network with an iterative algorithm like gradient descent, the general trend we see is that the cost function decreases up to a certain point after which it begins to increase, which is indicative of the model overfitting the data.

**Early stopping** involves stopping the training process when the error begins to increase.

#### Why early stopping works

Early stopping works because, during the beginning of the training process, the weights are close to zero, giving us a simple model.

As the training process increases these values and is eventually stopped, the value of the parameters will be relatively small, which leads to an effect similar to L2 regularization.

#### Downsides of early stopping

- One of the downsides of early stopping is **orthogonalization**. It is always good to have tools that focus on performing a single task at a time. early stopping combines the task of reducing the cost function and preventing overfitting.
- Early stopping already has an alternative, which is using L2 regularization and training the neural network extensively.

The advantage of early stopping is that we get to try our the training process for varying network sizes without having to practice L2 regularization which involves an extra hyperparameter (the regularization parameter $$\lambda$$).

# Normalizing the Input Features

**Normalization** is one of the methods which we can use to speed up the training process of out neural network.

It is used when the input features have highly differing scales which can lead to the cost function being distorted, hence causing the optimization algorithm's path to be inefficient.

## Steps to normalize a dataset

Consider a dataset with input features as shown

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/unnormalized_data.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

We aim to normalize this dataset this data such that it is suitable for our deep learning model to work on. The steps to do so are as follows.

- **Zero centering**: Here the mean of the features in question in training set is calculated and then subtracted from every instance of that feature in the training dataset. The resulting dataset has a **zero mean**.

$$ \text{mean }(\mu) = \frac{1}{m}\sum\limits_{i=1}^{m}X^{(i)}$$

$$\text{Zero centered data }(X) := X - \mu$$

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/zeroed_mean.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The example dataset with zero mean.
</div>

- **Normalizing the variances**: The variance $$(\sigma^2)$$ of the zero center dataset is now computed, using which we can normalize the data. The process is as follows:

$$\sigma^{2} = \frac{1}{m}\sum\limits_{i=1}^{m}X^{(i)}**2 \;\;(\text{ **} \Rightarrow \text{element-wise squaring})$$

$$X := \frac{X}{\sigma}$$

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/variances_normalized.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The dataset after normalizing the variances
</div>

When doing this, the same $$\mu$$ and $$\sigma^{2}$$ should be used to normalize the test set. This is because we want the training and test set to go through the exact same transformations defined by the same $$\mu$$ and $$\sigma^{2}$$ calculated on our training data.

## Reasons for Normalizing the Inputs

Without normalization, the cost function would resemble an elongated bowl as a result of features possessing extremely different scales.

Normalizing the input data will lead to the cost function being more uniform. This leads to the model being easier to train with relatively larger learning rates.

Normalizing the input features is crucial when the input features have extremely different ranges.

# Vanishing and Exploding Gradients

**Vanishing and Exploding Gradients** are some of the most common issues faced when training deep learning models. Here, the slopes/derivatives can either get very big or very small as the network gets more complex, making the training process difficult.

When training a neural network, with every subsequent activation, the outputs can either grow (explode) or reduce (shrink) exponentially due to the weight matrices being scaled slightly smaller or larger with each layer that is passed. The same is observed during the backpropagation process.

A potential solution to this is careful **weight initialization**.

## Weight Initialization in Deep Neural Networks

Taking a single neuron as an example with $$n$$ inputs such that the linear part is the weighted sum, ignoring the bias, given by

$$Z = W_{1}X_{1} + W_{2}X_{2}+ W_{3}X_{3}+ \dots + W_{n}X_{n}$$

To ensure that $$Z$$ does not blow up or become too small, the larger the $$n$$ is, the smaller the weights $$W_{i}$$ are.

In practice, using python this is done so as

```python
W_[l] = np.random.randn(shape)*np.sqrt(1/n[l-1])
```

where `n[l-1]` is the number of units from the (l-1)th layer being fed into the current layer.

A reasonable thing to do would be to set the variance of the weights to $$1/n$$ where $$n$$ is the number of input features going into the neuron.

For a ReLU activation function, a variance of $$2/n$$ works a bit better. This is done by taking a Gaussian random variable and multiplying it with $$\sqrt{2/n}$$, hence setting the variance to $$2/n$$.

If the input activations are roughly mean $$0$$ and standard variance $$1$$, $$Z$$ will also take on a similar scale, hence helping reduce the issue with vanishing and exploding gradients problem as it sets each of the weight matrices $$W$$ so that it is not too much bigger than 1 and not too less than 1, hence preventing vanishing or exploding too quickly.

### Other variants of random initialization

The previous section assumes the use of the ReLU activation function. However, in the case of other activation functions we use different values.

For the tanh activation function $$\sqrt{1/n}$$ is more suitable. This is called **Xavier Initialization**.

Another value proposed by Yoshua Bengio and his colleagues is

$$\sqrt{\frac{2}{n^{[l-1]} + n^{[l]}}}$$

These all give us a starting point for the variance of the initialization of the weight matrices.

The variance is another hyperparameter that can be tuned, which can have a moderate effect on the models' performance.

# Gradient Checking

**Gradient Checking** is a method used to verify that the gradients calculated during the backpropagation process are correct.

In this process we approximate the gradient of each parameter and tweak the value slightly to observe the change in the cost function. If the computed gradient and its approximation have close values, then the backpropagation process done is correct.

Consider a cost function where instead of the it being $$J(W^{[1]},b^{[1]}, \dots)$$, we have $$J(\theta)$$.

Similarly, the derivatives $$dW^{[1]}, db^{[1]}, dW^{[2]}, db^{[2]} \dots$$ can be reshaped into a big vector $$d\theta$$.

Now, the main question we have is whether $$d\theta$$ the slope of cost function $$J(\theta)$$. This is where gradient checking is useful.

The process is as follows

We first calculate

$$
\begin{align*}\\
&for \; each \; i:\\
&\;\;\;\; d\theta_{approx}[i] = \frac{J(\theta_{1},\theta_{2},\dots, \theta_{i}+\epsilon,\dots) - J(\theta_{1},\theta_{2},\dots, \theta_{i}-\epsilon,\dots)}{2\theta}
\end{align*}
$$

This gives us two vectors to work with, $$d\theta_{approx}$$ and $$d\theta$$, both of which have the same dimensions. We now check to see whether both these vectors are approximately equal to one another. This is mainly done through the Euclidean distance between the two

$$\frac{||d\theta_{approx} - d\theta||_{2}}{||d\theta_{approx}||_{2} + ||d\theta||_{2}}$$

In practice, we could use $$\epsilon=10^{-7}$$, and if the above gives us a value approximately equal to $$10^{-7}$$, our derivative approximation is very likely correct. If it is $$10^{-5}$$ or higher, we have to check if any components are too large and if some of the components' difference are very large, we might have a bug somewhere.

If the value obtained goes up to $$10^{-3}$$, then there is definitely a reason for concern. We will have to check the individual components to see whether a specific value of $$i$$ for which $$d\theta_{i}$$ is very small and use that to track down if some of our derivatives are incorrect.

## Practical Tips when working with Gradient Checking

- **Do not use grad checking during training**: Computing the approximated gradients $$d\theta_{approx}[i]$$ for every $$i$$ is computationally intensive. We only use this to confirm that our approximated gradients  are close to the actual backprop gradients $$d\theta$$ after which it is deactivated.
- **Debug the algorithm with component analysis**: If $$d\theta_{approx}$$ is very far from $$d\theta$$, we must analyze each of the components in our network to see which values of $$d\theta_{approx}$$ differ to a large extent from $$d\theta$$. This doesn't identify the bug right away but it helps us  home in on the location of the bug.
- **Regularization**: We must not forget the regularization term if we are regularizing our cost function during backpropagation. The gradients should be calculated for the entire cost function including the regularization term.
- **Issues working with Dropout**: Gradient checking will not work with dropout. This is because every iteration of dropout randomly eliminates different subsets of hidden units. As a result there is no easy cost function $$J$$ that the dropout can do gradient descent on. A good practice is to turn off dropout when performing gradient checking. We can also fix the pattern of nodes dropped and verify that grad check for the pattern of units dropped is correct.
- **The need for more training iterations**: We might see that backpropagation is more accurate when the weights and biases' values are close to zero. To counter this we have to run gradient checking for several iterations, allowing the weights and biases to move away from their initial values.

# Optimization Algorithms

Optimization algorithms are methods that allow us to train our neural networks faster.

Given that machine and deep learning are highly empirical, we have the need to train a lot of models to find  one that works well. Hence, it is helpful to be able to train models quickly. However this process is slow owing to the size of the datasets usually used when training deep learning models. Here we will be looking at some optimization methods that help with hastening the training process.

## Mini Batch Gradient Descent

In the real world, training neural networks is slow due to the sheer size of the dataset being worked with. Performing normal gradient descent on this whole dataset would not be feasible as it would take forever for even a single iteration to complete.

To combat this we divide the training set into smaller sets called **mini batches** and perform updates on each batch.

#### Implementing Gradient Descent

- Split the training data into mini batches of 1000 data points each such that
    - Mini batch 1: $$X_{1}, \dots, X_{1000}$$
    - Mini batch 2: $$X_{1001}, \dots, X_{2000}$$
    - and so on
- For each mini-batch perform the forward propagation process, keeping it isolated from the other batches
- Compute the cost function for the mini batch.
- Compute the gradients through backpropagation
- Update the weights and biases accordingly

#### Important points on mini batch gradient descent

- Mini batch gradient descent allows us to effectively train our model despite being partially through the dataset as one single pass takes multiple gradient descent steps.
- Every iteration of this process should lead to a decrease in the cost function.
- Different training batches may not lead to a decrease in the cost function due to various factors such as the the learning rate, the dataset itself being trained on and other such difficulties.

#### Behavior of Gradient Descent Algorithms

Mini batch size is one of the many hyperparameters we will have to consider when designing our neural network architecture.

When the size of the mini batch $$m$$ is equal to the size of the training set, then the process is known as **batch gradient descent**.

When the size of the mini batch is 1, then we end up with **stochastic gradient descent**.

<div class="row justify-content-center mt-3">
    <div class="col-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/contour.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Effects of different batch sizes on finding the minimum during the training.
</div>

Batch gradient descent would take relatively low noise and large steps when moving towards the minimum. However, it is also the slowest method owing to the fact that it has to process the whole dataset in one go at each iteration.

Stochastic gradient descent takes a gradient descent step with a single training example. Hence, most of the time we head towards the global minimum, but sometimes we can head in the wrong direction. As a result, it is an extremely noisy process but takes us in the correct direction eventually, while sometimes going in the wrong direction.

Stochastic gradient won't ever converge and will always oscillate around the region of the minimum.

#### Choosing mini batch size

For smaller training sets of less than 2000 examples, we use batch gradient descent.

For larger sets, sizes between 64-512 (powers of 2) are used to ensure that it is compatible with the memory layout of the computer.

We must always ensure that the mini batch size taken fits in the memory of the CPU or GPU being used.

# Exponentially Weighted Averages

There exist a large number of algorithms that can perform better than gradient descent. We will now look at a few of them, but before that, we need to know about the underlying concept they are based on, which is known as **exponentially weighted averages**.

Consider the temperature values in a town over a year. The data would normally be noisy. If we want to compute the trends then, as opposed to normal average, we have to make use of a _moving average_ of the temperature to keep track of the day to day changes.

If we took the initial weighted average to be $$v_{0}=0$$, and on every day, we average it with a weight of $$0.9$$ times the previous $$v$$ value that appears and $$0.1$$ times the given day's temperature, we would get the weighted average calculated as follows

$$\begin{align*}
v_{0} &= 0\\
v_{1} &= 0.9v_{0} + 0.1\theta_{1}\\
v_{2} &= 0.9v_{1} + 0.1\theta_{2}\\
\vdots
\end{align*}$$

The generalized formula for this is

$$v_{t} = 0.9v_{t - 1} + 0.1\theta_{t}$$

which is known as a moving average or an **exponentially weighted average** of the daily temperatures.

Considering the reading for a given day $$t$$ as $$\theta_{t}$$ and the weighted average for a given day as $$v_{t}$$, the general formula for weighted average is

$$\boxed{v_{t} = \beta v_{t - 1} + (1-\beta)\theta_{t}}$$

$$v_{t}$$ can be thought as approximately averaging over $$\frac{1}{1-\beta}$$ days' temperature.

For $$\beta = 0.9$$, we think of this process as averaging over the last 10 days' temperature.

Similarly, if we had $$\beta = 0.98$$, we would be averaging over, roughly the last 50 days.

Hence, as is seen below, for a high beta value the average curve (green) is smoother as we are averaging over more days of temperature. On the flip side, this curve is also shifted towards the right as we are averaging over a much larger temperature window. Hence, this exponentially weighted average formula adapts more slowly.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/year_temp.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Temperature distributions over the year with the weighted averages plotted at different values of beta
</div>

This leads to more latency as at a $$\beta$$ value of $$0.98$$ we are giving a lot of weight to the previous value and a smaller weight to the current value, hence leading to this average adapting more slowly.

If we set $$\beta=0.5$$, we would be averaging over just 2 days, giving us the yellow curve in the graph above which is much more noisy and susceptible to outliers, but also adapts much quickly to temperature changes.

Hence, the $$\beta$$ value here is yet another hyperparameter which can give us slightly different effects based on its value, and it is usually some value in between that would work best.

## Another explanation of weighted averages

The Exponentially Weighted Average (EWA) or Exponentially Weighted Moving Average (EWMA){% cite tobias2022 %} is used as a smoothing technique in time series data, and in this case, as a way to smooth out the cost function of neural networks when training them using mini batch gradient descent.

This algorithm depends on a _weight parameter_ $$\beta$$ which helps decide how much the current observation contributes to the total when calculating the EWA.

So taking the equation of exponential averages, we get the mean calculation as

$$
v_{t} = \beta v_{t - 1} + (1-\beta)\theta_{t}
$$

where
- $$v_{t}$$ is the EWA for the point of time/epoch $$t$$
- $$\theta_{t}$$ is the current reading at $$t$$
- $$\beta$$ is the weight parameter

The EWA combines the current reading being taken with the previous readings iteratively as (taking $$\beta=0.9$$)

$$
\begin{align*}
v_{1} &= 0.9v_{0}+0.1\theta_{1} \\
v_{2} &= 0.9v_{1}+0.1\theta_{2} \\
\vdots \\
v_{t} &= 0.9v_{t-1}+0.1\theta_{t} \\
\end{align*}
$$

Here
- $$\beta v_{t - 1}$$ is the **trend**
- $$(1-\beta)\theta_{t}$$ is the **current value**

A higher $$\beta$$ value places a higher importance on the trend than the newer values being added to it, leading to the curve adapting more smoothly.

A lower $$\beta$$ value would make the EWA place more importance on tracking newer values hence making the curve more noisy.

#### What is $$\beta$$?

$$\beta$$ can be thought of as

$$
n_{\beta} = \frac{1}{1-\beta}
$$

which is the number of observations used to adapt the EWA.
- For $$\beta=0.9$$, $$n$$ would be 10 leading to a **normally adapting** curve.
- For $$\beta = 0.5$$, $$n$$ would be 50 leading to a **slowly adapting** curve.
- For $$\beta = 0.98$$, $$n$$ would be 2 leading to a **quickly adapting** curve.

## Implementing Exponentially Weighted Averages

In pseudocode, the exponentially weighted average calculation process would be

$$
\begin{align*}
&V_{\theta} = 0 \\
&repeat\{ \\
&\text{get next } \theta_{t} \\
&\text{set } V_{\theta} := \beta V_{\theta} + (1-\beta)\theta_{t} \\
&\}
\end{align*}
$$

## Bias correction

Bias correction can help us make the calculations of the weighted average more accurate.

Taking the same example as before

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/year_temp.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Temperature distributions over the year with the weighted averages plotted at different values of beta
</div>

If we implement weighted average as

$$v_{t} = \beta v_{t-1} + (1 - \beta)\theta_{t}$$

we won't actually get the green curve in the plot above, we would instead get a curve looking more like the purple curve shown below.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Bias Correction.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

In this case the curve starts off very low.

This is because, when we implement the moving average, we initialize the value with $$v_{0}=0$$, taking $$\beta=0.98$$ and then calculate

$$v_{1} = 0.98v_{0}+0.02\theta_{1} \Rightarrow 0.02\theta_{1}$$

So, if the initial value is $$\theta_{0}=40$$, then $$v_{1}$$ will be $$0.02 \times 40$$ which is $$0.8$$. This is not a good estimate of what the initial value could be. Similarly $$v_{2}$$ would be

$$0.0196 \theta_{1} + 0.02\theta_{2}$$

which is again, not a good estimate.

To modify this estimate to make it more accurate, instead of taking $$v_{t}$$ we take

$$\frac{v_{t}}{1-\beta^t}$$

where $$t$$ is the current data point we are observing.

Considering the temperatures per day example from before, for $$t=2$$, $$1-\beta^{t} = 1-0.98^{2}= 0.0396$$. Considering the initial value to be $$40$$, we get the estimate of the temperature on day 2 as

$$
\frac{v_{2}}{0.0396} = \frac{0.0196\theta_{1} + 0.02\theta_{2}}{0.0396}
$$

Notice that the two known values on the top add up to the denominator. This gives us the weighted average of $$\theta_{1}$$ and $$\theta_{2}$$ and removes the bias.

As $$t$$ becomes large $$\beta^{t}$$ becomes close to $$0$$, so the bias correction makes almost no difference.

Hence, bias correction helps during the initial phases of learning when we are still warming up our estimates, to get a good estimate early on.

# Gradient Descent with Momentum

This is the first optimization algorithm we will be looking at which implements weighted averages.

Another name for this algorithm is **momentum**. This algorithm almost always works better than standard gradient descent.

Here, we calculate a weighted average of the gradients. This aggregates the gradients for a given number of epochs, say, 10 epochs, into a single weighted average value which is used in place of the actual gradient value to calculate the new value of the parameters.

This induces an effect of momentum where, when a new point is introduced to the given loss function, it won't contribute too much to the effective average value. This prevents particularly sharp increases or decreases in the gradient values from affecting the effective trajectory of the loss function.

As a result the overall gradient descent process is slowed down due to it preventing us from using a larger learning rate as it would diverge and overshoot the minimum.

Another way to think of this is _minimizing the bowl shape of the function_. Consider a ball rolling down a hill. The derivative terms act as acceleration on the ball rolling downhill and the momentum terms represent velocity of the ball. $$\beta$$ here acts as the friction preventing the ball from speeding up too much as the acceleration is imparted to it. As a result, rather than gradient descent taking every step independently of all the previous steps, the ball can gain momentum as a result of the aggregate velocity and the acceleration and friction acting on it.

Consider an example where we are optimizing a cost function with the following contour.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/contours.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

In the case of gradient descent, for every step, the cost function would zig zag till it gradually reaches the optimum.

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/contours_GD.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

One way to look at this is that we want to damp the oscillations in the vertical axis by slowing down the learning in that direction while hastening the learning process in the horizontal direction.

Each iteration of momentum goes through the following steps:

$$
\begin{align*}
&\text{On iteration t:}\\
&\;\;\;\;\text{Compute }dW,db\text{ on the current minibatch} \\
&\;\;\;\;V_{dW} = \beta V_{dW} + (1-\beta)dW \\
&\;\;\;\;V_{db} = \beta V_{db} + (1-\beta)db \\
&\;\;\;\;W = W - \alpha V_{dW} \\
&\;\;\;\;b = b - \alpha V_{db}
\end{align*}
$$

As a result the gradient descent steps are smoothed out

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/contour_momentum.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Advantages of Momentum

- Averaging out the gradients cause the oscillations in the vertical direction to move closer to zero.
- Oscillations in the horizontal direction, the derivatives, remain aggressive, moving towards the minimum.

As a result, the average in the horizontal direction is large.

## Implementing Momentum

The implementation of momentum is as follows

$$
\begin{align*}
&\text{On iteration t:}\\
&\;\;\;\;\text{Compute }dW,db\text{ on the current minibatch} \\
&\;\;\;\;V_{dW} = \beta V_{dW} + (1-\beta)dW \\
&\;\;\;\;V_{db} = \beta V_{db} + (1-\beta)db \\
&\;\;\;\;W = W - \alpha V_{dW}, \;\;b = b - \alpha V_{db}
\end{align*}
$$

This gives us two hyperparameters to work with: $$\alpha$$ and $$\beta$$.

The most common value for $$\beta$$ which controls the exponentially weighted average is $$\boxed{0.9}$$. That is averaging over the last 10 iterations' gradients.

The bias correction, is not usually used because after 10 iterations or so, our moving average would have warmed up and would no longer need a bias estimate.

In the above process $$V_{dW}$$ and $$V_{db}$$ are initialized to 0.

# RMSProp

RMSProp stands for **Root Mean Square Prop**. This is the second optimization algorithm we are looking at which also implements a moving average.

Taking the same example as before

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/contours.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Consider the vertical direction to be parameter $$b$$ and the horizontal direction to be parameter(s) $$W$$.

In RMSProp, we want to **slow down the learning** in the $$b$$ **vertical** direction and speed **up learning** in the $$W$$ **horizontal** direction.

## Implementing RMSProp

$$
\begin{align*}
&\text{On iteration t:}\\
&\;\;\;\;\text{Compute }dW,db\text{ on the current minibatch} \\
&\;\;\;\;S_{dW} = \beta_{2} S_{dW} + (1-\beta_{2})dW^{2}\text{ (the squaring operation here is element-wise)} \\
&\;\;\;\;S_{db} = \beta_{2} S_{db} + (1-\beta_{2})db^{2} \\
&\;\;\;\;W = W - \alpha\frac{dW}{\sqrt{S_{dW} + \epsilon}}, \;\;b = b - \frac{db}{\sqrt{S_{db} + \epsilon}}
\end{align*}
$$

This process keeps an exponentially weighted average of the **squares of the derivatives**.

Taking the weight updates here

$$W = W - \alpha\frac{dW}{\sqrt{S_{dW}+ \epsilon}}, \;\;b = b - \frac{db}{\sqrt{S_{db}+ \epsilon}}$$

We see that
- $$S_{dW}$$ is meant to be relatively small to speed up updates in the horizontal direction
- $$S_{db}$$ is meant to be relatively large to slow down updates in the vertical direction
- To ensure the algorithm doesn't divide by zero in practice we add a very small $$\epsilon$$ to the denominator. This enables better numerical stability and helps avoid dividing by zero and instead dividing by a very small value.

## Effects of RMSProp

Consider the image below

<div class="row justify-content-center mt-3">
    <div class="col-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Learning_directions.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Looking at the derivatives, the values are much larger in the vertical direction than the horizontal direction, that is, $$db$$ is much larger than $$dW$$ as the function has a steeper slope in the vertical direction.

This algorithm divides the updates in the vertical direction with a value that is a much larger number, helping damp out the oscillations.

The horizontal updates are divided by a smaller number. This leads to a net dampening of the updates.

As a result of this, **we can use a larger learning rate**, and get a faster learning **without the diverging in the vertical direction**.

RMSProp was proposed by a Coursera course by Jeff Hinton, not a paper.

# Adam Optimization

Adam optimization takes the ideas of Momentum and RMSProp and puts them together. It is widely used as it is robust and applicable to different deep learning architectures.

## Concepts used

The two algorithms used here are
- **Momentum**
    - This algorithm speeds up the optimization process by taking into account gradients from previous iterations.

    $$V_{dW} = \beta_{1} V_{dW} + (1-\beta_{1})dW$$

    $$V_{db} = \beta_{1} V_{db} + (1-\beta_{1})db$$

- **RMSProp**
    - This algorithm speeds up the optimization process by taking into account the moving average of the squared gradient.

    $$S_{dW} = \beta_{2} S_{dW} + (1-\beta_{2})dW^{2}$$

    $$S_{db} = \beta_{2} S_{db} + (1-\beta_{2})db^{2}$$

- **Bias Correction**
    - Bias correction is used in typical implementations of Adam. Look back to the [bias correction](#bias-correction) section to learn why this is important.
    - Bias correction for momentum

    $$V_{\text{dW_corrected}} = \frac{V_{dW}}{1-\beta_1^T}$$

    $$V_{\text{db_corrected}} = \frac{V_{db}}{1-\beta_1^T}$$

    - Bias correction for RMSProp

    $$V_{\text{dW_corrected}} = \frac{V_{dW}}{1-\beta_2^T}$$

    $$S_{\text{db_corrected}} = \frac{S_{db}}{1-\beta_2^T}$$

## Implementing Adam Optimization

$$
\begin{align*} \\
&V_{dW} = 0, \;S_{dW} = 0,\; V_{db} = 0, \;S_{db} = 0 \\
&\text{On iteration t:}\\
&\;\;\;\;\text{Compute }dW,db\text{ on the current minibatch} \\
&\;\;\;\;V_{dW} = \beta_{1} V_{dW} + (1-\beta_{1})dW,\;\; V_{db} = \beta_{1} V_{db} + (1-\beta_{1})db \\
&\;\;\;\;S_{dW} = \beta_{2} S_{dW} + (1-\beta_{2})dW^{2},\;\;S_{db} = \beta_{2} S_{db} + (1-\beta_{2})db^{2} \text{ ( element-wise squaring)} \\
&\;\;\;\;V_{dW}^{corrected} = V_{dW}/(1-\beta_{1}^{t}),\;\; V_{db}^{corrected} = V_{db}/(1-\beta_{1}^{t}) \\
&\;\;\;\;S_{dW}^{corrected} = S_{dW}/(1-\beta_{2}^{t}),\;\; S_{db}^{corrected} = S_{db}/(1-\beta_{2}^{t}) \\
&\;\;\;\;W = W - \alpha\frac{dW}{\sqrt{S_{dW}^{corrected}+\epsilon}}, \;\;b = b - \frac{db}{\sqrt{S_{db}^{corrected}+\epsilon}}
\end{align*}
$$

The hyperparameters here are
- $$\alpha$$: **The learning rate**. This value will need to be tuned
- $$\beta_1$$: **Momentum term**. This value is typically set to $$0.9$$
- $$\beta_2$$: **RMSProp term**. This value is typically set to $$0.999$$
- $$\epsilon$$: **used to avoid division by zero**. This value is typically set to $$10^{-8}$$

# Practically Implementing the Optimization Algorithms

A lot of the explanations for optimization algorithms tend to get confusing if only read. It's always good practice to see these in working. Here we will look at implementations of these learning algorithms from scratch and observe how they affect the learning process.

In this example we make use of the matyas function given by

$$f(x) = 0.26(x^{2}+y^{2})-0.48xy$$

which has its global minimum at $$(0,0)$$.

Before moving forward, we keep in mind the formulae for each of the optimization algorithms. These will be necessary to understand why these algorithms work.

| **Optimization Algorithm** | **Formula** |
|----------------------------|----------------------------|
| **Gradient Descent** | $$W = W - \alpha dW$$ <br> $$b = b - \alpha db$$ |
| **Momentum** | $$V_{dW} = \beta V_{dW} + (1-\beta)dW$$ <br> $$V_{db} = \beta V_{db} + (1-\beta)db$$ <br> $$W = W - \alpha V_{dW}$$ <br> $$b = b - \alpha V_{db}$$ |
| **RMSProp** | $$S_{dW} = \beta_2 S_{dW} + (1-\beta_2)dW^2$$ <br> $$S_{db} = \beta_2 S_{db} + (1-\beta_2)db^2$$ <br> $$W = W - \alpha \frac{dW}{\sqrt{S_{dW} + \epsilon}}$$ <br> $$b = b - \alpha \frac{db}{\sqrt{S_{db} + \epsilon}}$$ |
| **Adam** | $$V_{dW} = \beta_1 V_{dW} + (1-\beta_1)dW$$ <br> $$V_{db} = \beta_1 V_{db} + (1-\beta_1)db$$ <br> $$S_{dW} = \beta_2 S_{dW} + (1-\beta_2)dW^2$$ <br> $$S_{db} = \beta_2 S_{db} + (1-\beta_2)db^2$$ <br> $$V_{dW}^{corrected} = \frac{V_{dW}}{1-\beta_1^t}$$ <br> $$V_{db}^{corrected} = \frac{V_{db}}{1-\beta_1^t}$$ <br> $$S_{dW}^{corrected} = \frac{S_{dW}}{1-\beta_2^t}$$ <br> $$S_{db}^{corrected} = \frac{S_{db}}{1-\beta_2^t}$$ <br> $$W = W - \alpha \frac{dW}{\sqrt{S_{dW}^{corrected} + \epsilon}}$$ <br> $$b = b - \alpha \frac{db}{\sqrt{S_{db}^{corrected} + \epsilon}}$$ |

<br>

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/optimizer_demonstrations.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/optimizer_demonstrations.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

<br>

The animation generated from the above code is obtained as follows.

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/video/gradients_1.gif" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

Zooming in a bit, we can observe the loss curves a bit clearer as the approach the minimum.

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/video/gradients_2.gif" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

# Problems faced in Optimization

## Learning Rate Decay

One of the problems faced when optimizing our learning algorithm is reducing the learning rate over time, which is known as **learning rate decay**.

#### Reason for implementation

Mini batch gradient descent with small mini batches tends to be noisy when moving towards the minimum due to using a fixed learning rate value.

The solution for this is starting with a larger learning rate, which causes the algorithm to take to larger steps. We then eventually reduce the learning rate as we approach convergence.

#### Learning Rate Decay Methods
- Standard decay:
    - Formula: $$\alpha = \frac{1}{1+\text{decay Rate}\times \text{epoch number}}\alpha_{0}$$
    - The $$\alpha$$ values generated per epoch would look something like

| Epoch | $$\alpha$$ |
| ----- | -------- |
| 1     | 0.1      |
| 2     | 0.067    |
| 3     | 0.05     |
| 4     | 0.04     |

- Exponential Decay
    - Formula: $$\alpha = 0.95^{epoch}\alpha_{0}$$
    - The decay rate is much faster than standard decay due to use of an exponential scale
- Other Formulae
    - $$\boxed{\alpha = \frac{k}{\sqrt{ epoch }}\alpha_{0}}$$ where $$\alpha_{0}$$ is the initial learning rate value.
    - $$\boxed{\alpha = \frac{k}{\sqrt{ t }}\alpha_{0}}$$ where $$t$$ is the mini batch number.
- Discrete Learning Rates
    - We can make it so that the learning rate decreases in discrete steps.
- Manual Decay
    - If a model is taking too long, we can observe how the model is training and change the learning rate accordingly.

While learning rate decay can help speed up the training process, the primary learning rate has a more significant impact.

## Local Optima

When looking at cost function curves in the case of neural networks, most of the points of zero gradients are not local optima, they are in the form of **saddle points**.

At higher dimensional spaces, for a zero gradient in each direction, we can have either a convex-like function or concave-like function. For a several thousand dimensional space, we would get something like a saddle shaped curve.

**Plateaus** in these saddle curves can slow down the learning as it leads to the derivatives in those regions being close to zero for a very long time.

Optimization algorithms like RMSProp and Momentum are useful to help with this problem.

# Hyperparameter Tuning

Some of the common hyperparameters we work with when designing neural network architecture are:
- Learning rate $$(\alpha)$$
- Momentum term $$(\beta)$$
- Mini batch size
- Number of layers
- Number of hidden units
- Learning rate decay
- Adam optimizer parameters

## Order of importance of hyperparameters
- Learning rate is the most crucial parameter that we have to tune
- Secondary Importance:
    - Momentum Term
    - Mini batch size
    - Number of hidden units
- Tertiary Importance:
    - Number of layers
    - Learning Rate Decay
- Seldom tuned:
    - Adam optimizer parameters

## Hyperparameter sampling strategies

Earlier generations of ML algorithms used grids to systematically explore and choose the set of hyperparameter values that would work best.

For deep learning this is not ideal. We instead choose the hyperparameter value combinations at random. The reason for this is that we cannot predict what hyperparameters are important for our use case.

By sampling parameters at random we can explore a broad range of values for essential hyperparameters as opposed to a fixed set of values.

Another common practice is to use coarse to fine sampling schemes. Here we find a region where the parameters work well and then zoom in and sample more densely within the given region.

## Hyperparameter sampling at scale

When sampling hyperparameters at random, we do not randomly sample uniformly over all the possible values of the hyperparameters. The sampling is instead done at a scale.

### Uniform Sampling Examples

- When selecting the number of **hidden units** we use a value range of $$50$$ to $$100$$ as a starting point.
- Similarly for the **number of layers**, we use a value between $$2$$ to $$4$$.

### Sampling Learning Rate ($$\alpha$$)

- The sample range for learning rate is usually $$[0.0001,1]$$
- Uniform sampling over this range would be inefficient as most of the values would lie between $$0.1$$ to $$1$$.
- In this case we make use of a _logarithmic scale_.
- So, we take a logarithmic range between, for example, $$10^a$$ to $$10^b$$, selecting our random value on this range as $$10^r$$.

### Sampling Exponentially Weighted Average hyperparameters

- The hyperparameter $$\beta$$ is used for calculating exponentially weighted averages as has been seen [before](#exponentially-weighted-averages).
- The range of values used is in $$[0.9,0.999]$$.
- This case, similar to learning rate, is inefficient when being linearly sampled. Hence, we make use of logarithmic scale again.
- Here we explore the values of $$1-\beta$$, within a range of $$0.1$$ to $$0.001$$.
- We can take values of $$r$$ between $$-3$$ to $$-1$$ and subsequently find $$beta = 1-10^r$$.

We use the following sampling process because
- When taking $$\beta$$ values close to $$1$$, minor variations in the value can have huge impacts on the results obtained. For example
    - a change from $$0.9$$ to $$0.9005$$ has a minimal change
    - a change from $$0.999$$ to $$0.9995$$ drastically changes the results.
- the formula used when implementing bias correction, $$\frac{1}{1 - \beta}$$ is highly sensitive to minor changes when $$beta$$ is close to $$1$$.
- This sampling method allows for denser sampling as $$\beta$$ approaches $$1$$, hence allowing for efficient hyperparameter searching.

## Hyperparameter Tuning in Practice

Hyperparameter tuning is particularly important in deep learning especially due to the sheer number of hyperparameters involved in the field. In addition, intuitions from a given field of deep learning will not necessarily transfer to other fields.

Contrary to the above statement, there are some cases where we see some cross-fertilization between domains, such as ConvNets and ResNets in computer vision  finding use in speech synthesis in NLP. However, such cases are a few and far in between.

Intuitions on hyperparameters get stale over time due to several reasons such as
- Development of new algorithms for the same task
- Changes in data over time
- Infrastructure changes

Hence, it is necessary to re-evaluate our hyperparameters from time to time.

There are two main schools of thoughts that go into tuning hyperparameters
- **Babysitting a model** - this is done when we have a huge dataset and not a lot of computational resources.In this case we babysit the model even as it is training.
    - Here we begin with random parameter initialization and monitor the learning curve, adjusting the hyperparameters as per the need.
    - Changes are made incrementally over a period of time and the impacts on the model performance are actively observed.
    - In the case of the changes performed leading to the detriment of the models' performance, we revert back to a previous configuration
    - This is an long and intensive process.
- **Training many models in parallel** - This method is useful when we have sufficient computational resources to work with.
    - Here, we run multiple models simultaneously with different parameter settings.
    - Each model generates its own learning curve based on which we can make a decision on the best model to select

Making a decision between these approaches boils down to the computational resources we have at hand as well as the size of the datasets we are dealing with.

# Batch Normalization

**Batch Normalization** is one of the most crucial ideas in the field of machine learning and deep learning. This process involves calculating the means and variances of the dataset and adjusting the data based on them. The end result is transforming our problem at hand to make it more suitable to be a gradient descent optimization problem.

Batch normalization is meant to help make the parameter search process easier, hence making the neural network more robust. It also enables the training of very deep neural networks.

## Batch Normalization Mechanism
As opposed to normalizing the activations, batch normalization focuses on normalizing the linear part $$z$$ before the activation function. The process is as follows:
- Compute the mean as: $$\mu = \frac{1}{m} \sum\limits_{i}Z^{(i)}$$.
- Compute the variance: $$\sigma^{2} = \frac{1}{m} \sum\limits_{i}(Z^{(i)} - \mu)^{2}$$
- Normalize the values: $$z^{(i)}_{norm} = \frac{Z^{(i)}-\mu}{\sqrt{ \sigma^{2} + \epsilon }}$$
- Scale and shift the normalized values: $$\tilde{Z}^{(i)} = \gamma Z^{(i)}_{norm} + \beta$$

Here $$\gamma$$ and $$\beta$$ control the mean and variance of the hidden units. Setting $$\beta = \mu$$ and $$\gamma = \sqrt{sigma^{2} + \epsilon}$$, we get the identity function $$z^{(i)}_{norm}$$.

This process ensures that the input and hidden units have a standardized mean and variance.

## Batch Normalization in Deep Neural Networks

Batch normalization is used to improve the generalization, accelerate training and allow for the use of higher learning rates in deep neural networks.

Batch normalization is applied to each mini batch as opposed to the whole training set

## Implementation
For each mini batch
- Compute the mean and variance of the activations ($$Z$$) alongside the normal forward propagation process.
- Normalize the activations $$Z^{[l]}$$ using the statistical information calculated previously to get $$\tilde{Z}^{[l]}$$.
- Scale and shift the normalized activations using the two parameters $$\beta$$ and $$\gamma$$
- Use backpropagation to compute $$dW^{[l]},d\beta^{[l]},d\gamma^{[l]}$$
- Update the parameters

Batch normalization replaces the role of the biases $$B$$ with $$\beta$$ and also introduces a scaling factor $$\gamma$$. Hence the parameters we are working with are $$W,\beta,\gamma$$.

This process can be used for other optimization methods like RMSProp, and momentum.

## Advantages

- Batch normalization helps deal with covariate shift{% cite Seldon2021 %}.
- It ensures that all the parameters have similar value ranges while maintaining the variances. This makes the training process more efficient.
- It centers the activations around zero mean due to which the bias term becomes redundant. Due to this they can be set to zero or removed.

The $$\beta$$ in batch normalization is not the same as the $$\beta$$ in other algorithms such as momentum or Adam.

## Why Batch Normalization Works

We have already seen how [normalization](#normalizing-the-input-features) helps improve the speed performance and stability of neural networks.

Some important points about batch normalization are

- **Feature Scaling**: Scaling down our features to have a mean of zero and a variance of one, as is known beforehand, can accelerate the learning process. In batch normalization, this concept is extended to the hidden layers.
- **Robustness to weight changes**: Changes in the earlier layers of a neural network can have a significant effect on the values of the weights on the deeper layers of the network. Batch normalization helps make the network more resilient to this.
- **Covariate Shift**: When the distribution of the data changes, it will affect the model performance, hence requiring the model to be retrained. This is known as covariate shift. In neural networks, as the early layer weights undergo changes, the values in the subsequent layers get affected, hence leading to an internal covariate shift. Batch normalization helps reduce this effect.
- Batch normalization makes it such that even if values from previous layers change, their mean and variance remain consistent, hence making the network stable. As a result each layer can learn independently, accelerating the learning process across the whole network.
- Batch norm also has a slight regularization effect due to the introduction of noise when computing the mean and variances based on mini batches. This prevents the model from relying too much on any particular hidden unit. This effect is not the main effect we are trying to find here which is why it is not given as much importance. Normalization along with dropout is preferred to this.

## Batch Normalization During Testing

Batch normalization handles the data in mini batches, computing the means and variances on each mini batch.

Due to this, when making predictions and evaluating the neural network, we have one example at a time instead of mini batches. Hence, we need to do something to ensure that our predictions are accurate.

The training process is as follows:
- Calculate $$\sigma^2$$ and $$\mu$$

$$
\mu = \frac{1}{m}\sum\limits_{i}z^{(i)}
$$

$$
\sigma^{2} = \frac{1}{m}\sum\limits_{i}(z^{(i)} - \mu)^{2}
$$

- Normalize the activations using $$\sigma^2$$ and $$\mu$$

$$z^{(i)}_{norm} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^{2}+\epsilon}}$$

- Scale and shift the normalized activations using the parameters $$\gamma$$ and $$\beta$$

$$
\tilde{z}^{(i)} = \gamma z^{(i)}_{norm} +\beta
$$

The issue here is that we are calculating $$\sigma^2$$ and $$\mu$$ on a mini batch, but we might not have a mini batch to work with when testing.

To solve this:
- During training we first **maintain an exponentially weighted average** for $$\mu$$ and $$\sigma^2$$.
- Then, when it comes to testing, we normalize the activations with these averaged values and apply the $$\gamma$$ and $$\beta$$ learned during the training to shift and scale the data as necessary.

The advantage here is that the estimations for $$\mu$$ and $$\sigma^2$$ are robust and there are many other ways to approximate these values besides an exponentially weighted average.

# Multi-class Classification - Softmax Regression

The focus till now has been on binary classification through the use of logistic regression where we only have two possible outcomes to a prediction.

Real world situations, however, will require for us to classify much more than two objects simultaneously. This is where we use **Softmax Regression**.

Softmax is a generalization of Logistic Regression which handles classifying an unknown object to one of $$C$$ classes. Here we use $$C$$ to denote the number of classes we are trying to categorize the inputs to.

## Softmax in Neural Networks

In a neural network, the output layer will have $$C$$ output units. So, $$n^{[L]} = C$$. Each of the output nodes will give us the probability of the given input belonging to each of the $$C$$ classes.

The dimensions of the output is a $$(C,1)$$ vector with each of the nodes being $$P(C_{1}\mid X),P(C_{2}\mid X), \dots$$, all of them summing up to $$1$$.

To achieve this, the **softmax** layer comes into the picture.

When computing the softmax activation on the last layer, we first compute $$Z^{[L]}$$ which is the product of the weights and activations of the previous layer with the biases added to them.

We then calculate

$$t = e^{Z^{[L]}}$$

This is also a $$(C,1)$$ dimension vector. This is then fed into the activation, giving us the output $$a^{[L]}$$

$$\boxed{a^{[L]} = \frac{e^{Z^{[L]}}}{\sum_{i=1}^{C}t_{i}}}$$

For the $$i^{th}$$ element of the output this would be

$$\boxed{a^{[L]}_{i} = \frac{t_{i}}{\sum_{i=1}^{C}t_{i}}}$$


From this we see that the softmax activation function is used to transform a vector of numbers to a probability distribution for each class the outputs represent.

## How Softmax Works

Suppose we have a vector

$$
Z^{[L]} = \begin{bmatrix}
5 \\
2 \\
-1 \\
3
\end{bmatrix}
$$

we would get $$t$$ as

$$
t = \begin{bmatrix}
e^{5} \\
e^{2} \\
e^{-1} \\
e^{3}
\end{bmatrix} = \begin{bmatrix}
148.4 \\
7.4 \\
0.4 \\
20.1
\end{bmatrix}
$$

which is the exponent of the elements of the vector.

Now, if we wanted to get $$a^{[L]}$$, we normalize these values so that they sum up to 1, hence giving us what is effectively a probability distribution. Hence we calculate this as

$$\sum\limits_{j=1}^{4}t_{j} = 176.3$$

We then obtain the activation for this layer $$a^{[L]}$$ as

$$a^{[L]} = \frac{t}{176.3}$$

## Training the Softmax Classifier

Taking the same example as before, where we have

$$
Z^{[L]} = \begin{bmatrix}
5 \\
2 \\
-1 \\
3
\end{bmatrix}
$$

we calculate the softmax activation as

$$
g^{[L]}(Z^{[L]}) = \begin{bmatrix}
\frac{e^{5}}{e^{5} + e^{2} + e^{-1} + e^{3}} \\
\frac{e^{2}}{e^{5} + e^{2} + e^{-1} + e^{3}} \\
\frac{e^{-1}}{e^{5} + e^{2} + e^{-1} + e^{3}} \\
\frac{e^{3}}{e^{5} + e^{2} + e^{-1} + e^{3}}
\end{bmatrix} = \begin{bmatrix}
0.842 \\
0.042 \\
0.002 \\
0.114
\end{bmatrix}
$$

Here we see that the biggest vector is,5, due to which it has the highest probability value of the bunch.

Softmax is named as such because of its contrast to the **Hardmax** function which just shows the highest value to have a probability of $$1$$ with the rest of the values being set to $$0$$. Softmax, instead, is a gentle mapping of the $$Z$$ values to different probabilities, hence giving us a probability distribution as the output.

## Softmax Cost Function

The loss function of the softmax function given the ground truth as $$y$$ and the model prediction as $$\hat{y}$$, is

$$L(\hat{y},y) = -\sum\limits_{j=1}^{C}y_{j}\log{\hat{y}_{j}}$$

where $$C$$ is the number of possible classes that the output can have.

The loss function here is known as the **cross entropy loss function**, which looking at the ground truth, tries to make the corresponding probability of the class as high as possible.

Hence, the cost function would be

$$\boxed{J(W^{[I]},B^{[1]},\dots) = \frac{1}{m}\sum\limits_{i=1}^{m} L(\hat{y}^{(i)},y^{(i)})}$$

## Conclusion

The methods discussed in this blog are some of the most popular methods used to improve the training speed of our neural networks along with  in several ways.

We have looked into
- Appropriate ways to split up our dataset when dealing with deep learning problems and how it differs from classical machine learning.
- Ways to prevent the model from overfitting the data through regularization methods such as L2, L1 and Dropout.
- Normalizing our datasets to help our deep learning models to learn faster.
- How appropriate weight initialization methods can help with vanishing and exploding gradients.
- Verifying that our calculated gradients are correct through gradient checking.
- Optimization algorithms other than simple gradient descent that use exponentially weighted averages to train our neural networks faster.
- Hyperparameter sampling strategies to help choose our deep learning hyperparameters effectively.
- Batch Normalization for deep neural networks and how they help prevent instability in the weights and biases when training deeper neural networks.
- Multi-class classification with the help of the **softmax** activation function.

Now that we have looked into some of the ways we can improve our deep learning models, We can now shift focus to deep learning model architectures that are more tailored towards specific applications which would be beyond the capabilities of simple deep artificial neural networks.