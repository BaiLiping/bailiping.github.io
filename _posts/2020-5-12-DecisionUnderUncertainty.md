---
layout: post
title: Decision Making Under Uncertainty 
subtitle: How Really Decisions Are Made
tags: [rl]
---
Reinforcement Learning is more or less made on the premise that you know all the choices and there is a way to descern how good or bad that choice is. Even for a n step problem, so long you have a way to assess that distribution for every step, then you can recurssively build a strategy table by reasoning from the last step. 

There is that topic of Hidden Markov Model, which is a topic I need to pick up more on. For this particular discussion, we should focus on the most generic verion of RL.

That desicion making process rely on two crutial assumptions:
1. You know all the possible scenarios and the distribution associated with it.
2. You know the choices and the stochastity associated with it.

I think there is a sociatal analog to that process. We observe people before us and our peers make decisions under various conditions and how they are rewarded for their decisions. Maybe through that kind of observation, we have an action reward table.  


Here is a list of rule of thumbs that I can think of to guide us in decisions under uncertainty:

1. Prepare for the worst case scenario. 
2. Secondary, Tertiary thinking. A good example of this would be this covid stuff. The disease manifestation is rarely from virus, but an overly excited immune system. Other than anti-virual, one treatment option would be immune surpressent drugs such as IL6 inhibitor. Therefore, one clear secondary coneqence of this would be that other bacteria and virus that are at equilibrium with the immune system. Now with the immume system surpressed to treat the covid stuff, those bacterias would sneak out. This is a typical secondary effect of an action. Another example would be riots. When things first started to pick up in US, a lot Chinese people decided to go back to China. At that point, there are people who made fun of their choices since the ticket price is off the roof. However, in retrospect, that might be the right choice. Not so much for the COVID stuff itself, but the riots and anti-asian stuff that are almost inevitable. 
