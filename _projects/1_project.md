---
layout: page
title: Deep Learning Optimization Algorithm Demonstrations
description: Here I go a bit in depth on how some of the most well known deep learning optimization algorithms work
img: assets/project_bgs/project1_bg.jpg
importance: 1
category: work
related_publications: true
---

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/filter_demonstrations.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/project_files/optimizer_demonstrations.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}