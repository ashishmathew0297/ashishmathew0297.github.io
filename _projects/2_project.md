---
layout: page
title: Recreating ResNet-34
description: Following the research paper from 2015, I recreate the ResNet-34 model from scratch using PyTorch 
img: assets/project_bgs/project2_bg.jpg
importance: 1
category: work
related_publications: false
references: false
---

[Hallo everynyan! How are you? Fine thank you.](https://youtu.be/nlLhw1mtCFA) Here's my rendition of a rather interesting CNN architecture I read up on during my self learning of DL. I prefer to keep things light hearted because of how... you know what, nevermind.

Given my exploration of CNNs from scratch going through different papers and such, I thought it would be a good idea to try recreating it from scratch.

The original project is in [this link](https://github.com/ashishmathew0297/ResNet34), but I have not included the dataset itself due to GitHub's limitations.

This is the first model to introduce the concept of skip connections that find use in a lot of modern day networks, and act as a sort of data bridge from earlier layers of a network to later layers. This helps make the training process easier as it gives later layers access to outputs from earlier layers, to help avoid saddle points in the loss function calculation. 

The link to the original paper: [Here ya go](https://arxiv.org/abs/1512.03385)

I have also personally been interested in exploring RNNs in the near future and I hope to be able to clearly explain what is happening under the hood for these models.

{::nomarkdown}
{% assign jupyter_path = "assets/project_files/my_version.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/project_files/my_version.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}
