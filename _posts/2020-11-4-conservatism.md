---
layout: post
title: Minimizing KL Divergence as a Form of Conservatism
subtitle: At the Boundary between The Old Distribution and The New Distribution
tags: rl
image:
show-avatar: false
social-share: false
comments: true
---

Conservatism is a style of thinking: Cicrumstances changes, there might be new objectives pops up. THe old way of doing things might no longer be compatable to reaching the desired objective, so what are the guiding principles as we make changes? A more drastic style is to just abandom the old and build around the new. However, while it might sounds good, there are drawbacks. What if the seemingly good new has potential downside that we can yet understand? A more reasonable aproach is to acknowledge the wisdom in a system that already works, and seek to propose changes in order the reach the target under current system. One constraint would be to minimize the KL divergence. While we make changes, we make sure that the new distribution in order to reach a new goal, we also want to we minimoize the distance to the old one such that the old system is preserved to the largest extend.

There is also a paradox when it comes to rathional decision making. Let us say that there are two preference distributions, one is the your current decision distribution, another is the one of a mathematician. The two distributions give different weights to choices. When you don't have that curve of mathematicians, you would never make the choice the same as his. But let us suppose that somehow you want to be a mathematician, so how can that choice be a rational one if you don't have access to understand how good it is to be a mathematician? In this way of looking at things, the decision to be a mathematician would be a irrational one.

I think contingency table can help here. Let us suppose that while I don't have the exact distribution of a mathematician, but i am fond of them, so i suppose that the upside is 100. The downside of becoming a mathematician is the time one spent, say -10 points. then i assign the probability of how likely i would become a mathematician if i spend the time one it, and would get a utility score. IF that score is better than the score of my current path, then i would start the process of updating my distribution into that of a mathematician.

There is still a sense of distance here. The new curve has to be close enough to my old curve such that I can do the contingency analysis on it. 
