---
layout: post
published: true
title: 'Recurrent and Recursive Nets: Part 1'
subtitle: >-
  Introduction to recurrent networks and unfolding computational graphs from the
  Chapter 10 of the Deep Learning book with a YouTube lightboard lecture
---
<iframe width="560" height="315" src="https://www.youtube.com/embed/jEfGXrnfSWQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## 10.0 Introduction
Recurrent neural networks, or RNNs are a family of neural networks for learning from sequential data, i.e., $\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(\tau)}$.
In practice, they are usually operating on mini-batches of such sequences. However, we omit them for notional simplicity.

To achieve this, we utilize the "parameter sharing" idea to apply a model to examples of variable lengths and generalize across them. This is crucial if a particular piece of information could occur at any position in the sequence.

Another related approach is the use of 1-D convolution across a temporal sequence which is the basis of time-delay networks. However, this approach shares parameters across a short time-span.

RNN's extend the idea of computational graphs to include cycles which represent the influence of the present value of a variable to its future values.

## 10.1 Unfolding Computational Graphs
In this section, we explain the idea of unfolding recursive or recurrent computation into a computational graph that has a repetitive structure.
For example, consider the classical form of a dynamical system:

$$
s^{(t)}=f\left(s^{(t-1)} ; \theta\right)
$$

Such an expression can now be represented by a traditional directed acyclic computational graph like this,
![Classical Dynamical Systems.png]({{site.baseurl}}/assets/img/rnn/Classical Dynamical Systems.png)

if we unfold equation this for $\tau = 3$ time steps, we obtain,

$$
\begin{aligned}
\boldsymbol{s}^{(3)} &=f\left(\boldsymbol{s}^{(2)} ; \boldsymbol{\theta}\right) \\
&=f\left(f\left(\boldsymbol{s}^{(1)} ; \boldsymbol{\theta}\right) ; \boldsymbol{\theta}\right)
\end{aligned}
$$

![fig10-2.png]({{site.baseurl}}/assets/img/rnn/fig10-2.png)

As another example, let us consider a dynamical system driven by an external signal

$$
\boldsymbol{h}^{(t)}=f\left(\boldsymbol{h}^{(t-1)}, \boldsymbol{x}^{(t)} ; \boldsymbol{\theta}\right)
$$

typical RNNs will add extra architectural features such as output layers that read information out of the state $\boldsymbol{h}$ to make predictions.

When the recurrent network is trained to perform a task that requires predicting the future from the past, the network typically learns to use $\boldsymbol{h}(t)$ as a kind of lossy summary of the task-relevant aspects of the past sequence of inputs up to $t$. This summary is in general necessarily lossy, since it maps an arbitrary length sequence $\left(\boldsymbol{x}^{(t)}, \boldsymbol{x}^{(i-1)}, \boldsymbol{x}^{(i-2)}, \ldots, \boldsymbol{x}^{(2)}, \boldsymbol{x}^{(1)}\right)$ to a fixed length vector $\boldsymbol{h}^{(t)}$.

One way to draw this RNN is with a diagram containing one node for every component that might exist in a physical implementation of the model, such as a biological neural network. In this view, the network deﬁnes a circuit that operates in real-time, with physical parts whose current state can inﬂuence their future state, as in the left of ﬁgure bellow figure.T throughout this chapter, we use a black square in a circuit diagram to indicate that an interaction takes place with a delay of a single time step, from the state at time $t$ the state at time $t+ 1$. The other way to draw the RNN is as an unfolded computational graph, in which each component is represented by many diﬀerent variables, with one variable per time step, representing the state of the component then. Each variable for each time step is drawn as a separate node of the computational graph.

We can represent the unfolded recurrence after $t$ steps with a function $g^{(t)}$:

$$
\begin{aligned}
\boldsymbol{h}^{(t)} &=g^{(t)}\left(\boldsymbol{x}^{(t)}, \boldsymbol{x}^{(t-1)}, \boldsymbol{x}^{(t-2)}, \ldots, \boldsymbol{x}^{(2)}, \boldsymbol{x}^{(1)}\right) \\
&=f\left(\boldsymbol{h}^{(t-1)}, \boldsymbol{x}^{(t)} ; \boldsymbol{\theta}\right)
\end{aligned}
$$

The unfolding process thus introduces two major advantages:

* Regardless of the sequence length, the learned model always has the same input size, because it is speciﬁed in terms of the transition from one state to another state, rather than speciﬁed in terms of a variable-length history of states.
* It is possible to use the same transition function with the same parameters every time step.

These two factors make it possible to learn a single model of $f$ that operates on all the time steps and all sequence lengths.

## Reference
Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.
