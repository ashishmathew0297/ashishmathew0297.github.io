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

The original project is in [this link](https://github.com/ashishmathew0297/ResNet34)

Given my exploration of CNNs from scratch going through different papers and such, I thought it would be a good idea to try recreating it from scratch.

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
