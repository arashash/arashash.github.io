---
layout: post
published: true
title: 'Recurrent and Recursive Nets: Part 2'
subtitle: >-
  We continue the previous lesson by introducing different RNN architures and
  back-propagation through-time and teacher-forcing algorithms for training.
---
<iframe width="560" height="315" src="https://www.youtube.com/embed/C08mT2VSHGg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## 10.2 Recurrent Neural Networks
Now with the graph-unrolling and parameter-sharing ideas of the previous section, we can design a wide variety of recurrent neural networks. Let's consider the following 3 groups:

* Recurrent networks with recurrent connections between hidden units, that read an entire sequence and then produce a single output

Such a network can be used to summarize a sequence and produce a ﬁxed-size representation used as input for further processing:
![fig10-5.png]({{site.baseurl}}/assets/img/rnn/fig10-5.png)


* Recurrent networks that produce an output at each time step and have recurrent connections between hidden units:
![fig10-3.png]({{site.baseurl}}/assets/img/rnn/fig10-3.png)
for example, we can apply the following update equations

$$
\begin{aligned}
\boldsymbol{a}^{(t)} &=\boldsymbol{b}+\boldsymbol{W} \boldsymbol{h}^{(t-1)}+\boldsymbol{U} \boldsymbol{x}^{(t)} \\
\boldsymbol{h}^{(t)} &=\tanh \left(\boldsymbol{a}^{(t)}\right) \\
\boldsymbol{o}^{(t)} &=\boldsymbol{c}+\boldsymbol{V} \boldsymbol{h}^{(t)} \\
\hat{\boldsymbol{y}}^{(t)} &=\operatorname{softmax}\left(\boldsymbol{o}^{(t)}\right)
\end{aligned}
$$

Then we can calculate the loss like this,

$$
\begin{aligned}
& L\left(\left\{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(\tau)}\right\},\left\{\boldsymbol{y}^{(1)}, \ldots, \boldsymbol{y}^{(\tau)}\right\}\right) \\
=& \sum_{t} L^{(t)} \\
=&-\sum_{t} \log p_{\text {model }}\left(y^{(t)} |\left\{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(t)}\right\}\right)
\end{aligned}
$$

The back-propagation algorithm applied to the cost is called back-propagation through time(BPTT) and will be discussed in detail in the next lecture.

* Recurrent networks that produce an output at each time step and have recurrent connections only from the output at one time step to the hidden units at the next time step
![fig10-4.png]({{site.baseurl}}/assets/img/rnn/fig10-4.png)

Because this network lacks hidden-to-hidden recurrence, it requires that the output units capture all the information about the past that the network will use to predict the future.


### 10.2.1 Teacher Forcing and Networks with Output Recurrence
Models that have recurrent connections from their outputs leading back into the model may be trained with teacher forcing, in which during training the model receives the ground truth output $y(t)$ as input at time $t+ 1$.

The disadvantage of strict teacher forcing arises if the network is going to be later used in a closed-loop mode, with the network outputs (or samples from the output distribution) fed back as input. In this case, the fed-back inputs that the network sees during training could be quite diﬀerent from the kind of inputs that it will see at test time.

![]({{site.baseurl}}/assets/img/rnn/fig10-6.png)


## Reference
Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.
