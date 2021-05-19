---
layout: post
title: Fundamental of RL
tags: rl
image:
show-avatar: false
social-share: false
comments: true
---

<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

## Section 1: Policy Evaluation

You're now ready to begin the assignment! First, the city council would like you to evaluate the quality of the existing pricing scheme. Policy evaluation works by iteratively applying the Bellman equation for $$v_{\pi}$$ to a working value function, as an update rule, as shown below.

$$\large v(s) \leftarrow \sum_a \pi(a | s) \sum_{s', r} p(s', r | s, a)[r + \gamma v(s')]$$
This update can either occur "in-place" (i.e. the update rule is sequentially applied to each state) or with "two-arrays" (i.e. the update rule is simultaneously applied to each state). Both versions converge to $$v_{\pi}$$ but the in-place version usually converges faster. **In this assignment, we will be implementing all update rules in-place**, as is done in the pseudocode of chapter 4 of the textbook.

We have written an outline of the policy evaluation algorithm described in chapter 4.1 of the textbook. It is left to you to fill in the `bellman_update` function to complete the algorithm.

```python
def bellman_update(env, V, pi, s, gamma):
    v = 0
    for a in env.A:
        transitions = env.transitions(s, a)
        for s_, (r, p) in enumerate(transitions):
            v += pi[s][a] * p * (r + gamma * V[s_])
    V[s] = v
```

```python
def evaluate_policy(env, V, pi, gamma, theta):
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta: #if the improvement becomes too small, abandand update
            break
    return V
```

## Section 2: Policy Iteration
Policy iteration works by alternating between evaluating the existing policy and making the policy greedy with respect to the existing value function. We have written an outline of the policy iteration algorithm described in chapter 4.3 of the textbook. We will make use of the policy evaluation algorithm you completed in section 1. It is left to you to fill in the `q_greedify_policy` function, such that it modifies the policy at s to be greedy with respect to the q-values at s, to complete the policy improvement algorithm.

```python
def q_greedify_policy(env, V, pi, s, gamma):
    q = np.zeros_like(env.A, dtype=float) #because this is an array of float
    for a in env.A:
        transitions = env.transitions(s, a)
        for s_, (r, p) in enumerate(transitions):
            q[a] += p * (r + gamma * V[s_])   
    greed_actions = np.argwhere(q == np.amax(q))
    for a in env.A:
        if a in greed_actions:
            pi[s, a] = 1 / len(greed_actions)
        else:
            pi[s, a] = 0
```

```python
def improve_policy(env, V, pi, gamma):
    policy_stable = True
    for s in env.S:
        old = pi[s].copy()
        q_greedify_policy(env, V, pi, s, gamma)
        if not np.array_equal(pi[s], old):
            policy_stable = False
    return pi, policy_stable
```

```python
def policy_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    policy_stable = False
    while not policy_stable:
        V = evaluate_policy(env, V, pi, gamma, theta)
        pi, policy_stable = improve_policy(env, V, pi, gamma)
    return V, pi
```

## Section 3: Value Iteration
Value iteration works by iteratively applying the Bellman optimality equation for $$v_{\ast}$$ to a working value function, as an update rule, as shown below.

$$\large v(s) \leftarrow \max_a \sum_{s', r} p(s', r | s, a)[r + \gamma v(s')]$$
We have written an outline of the value iteration algorithm described in chapter 4.4 of the textbook. It is left to you to fill in the `bellman_optimality_update` function to complete the value iteration algorithm.

```python
def bellman_optimality_update(env, V, s, gamma):
    vmax = - float('inf')#starting from negative infinity
    for a in env.A:
        transitions = env.transitions(s, a)
        v = 0
        for s_, (r, p) in enumerate(transitions):
            v += p * (r + gamma * V[s_])
        vmax = max(v, vmax)
    V[s] = vmax
```

```python
def value_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    for s in env.S:
        q_greedify_policy(env, V, pi, s, gamma)
    return V, pi
```
