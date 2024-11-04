---
layout: page
title: Optimization Algorithm Demonstrations
description: Here I go a bit in depth on how some of the most well known deep learning optimization algorithms work
img: assets/project_bgs/project1_bg.jpg
importance: 1
category: work
related_publications: false
references: false
---

[Here's](https://github.com/ashishmathew0297/optimization-algorithms) the link to the original notebook for you to tinker with. Some of the latex formulae will not render properly on this page due to compatibility issues with Mathjax. Running the notebook locally or viewing it on github will render the formulae properly.

A slightly shorter version of this project has been included in one of the blogposts, however I decided to put it up here as well to allow you to see the code working by itself.

Before you go ahead, just for reference:

- The Matyas function: 

$$f(x,y) = 0.26(x^{2}+y^{2}) - 0.48xy$$

- The minimum of the Matyas function is at $$(0,0)$$

{::nomarkdown}
{% assign jupyter_path = "assets/project_files/optimizer_demonstrations.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/project_files/optimizer_demonstrations.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

The animations that I generated from the above code are shown below:

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/video/gradients_1.gif" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>

<div class="row justify-content-center mt-3">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/video/gradients_2.gif" class="img-fluid rounded z-depth-1" zoomable=true%}
    </div>
</div>