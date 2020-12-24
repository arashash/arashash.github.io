---
layout: post
title: Lovász-Softmax Loss for Semantic Image Segmentation
date: '2020-01-07 17:30'
excerpt: This blog post will guide you on how to understand and implement the lovasz-softamx loss function for image segmentation in Pytorch.
comments: true
---

*Definition 1. A set function $\Delta:\{0,1\}^{p} \rightarrow \mathbb{R}$ is submodular if for all $\mathbf{A}, \mathbf{B} \in\{0,1\}^{p}$

$$
\Delta(\mathbf{A})+\Delta(\mathbf{B}) \geq \Delta(\mathbf{A} \cup \mathbf{B})+\Delta(\mathbf{A} \cap \mathbf{B})
$$

The convex closure of submodular set functions is tight and computable in polynomial time; it corresponds to its Lovász extension.

*Definition 2. The Lovász extension of a set function $\Delta:\{0,1\}^{p} \rightarrow \mathbb{R}$ such that $\Delta(\mathbf{0})=0$ is defined by

$$
\begin{array}{c}
{\bar{\Delta}: \boldsymbol{m} \in \mathbb{R}^{p} \mapsto \sum_{i=1}^{p} m_{i} g_{i}(\boldsymbol{m})} \\
{\text {with } g_{i}(\boldsymbol{m})=\Delta\left(\left\{\pi_{1}, \ldots, \pi_{i}\right\}\right)-\Delta\left(\left\{\pi_{1}, \ldots, \pi_{i-1}\right\}\right)}
\end{array}
$$

$\pi$ being a permutation ordering the components of $\boldsymbol{m}$ in decreasing order, i.e. $x_{\pi_{1}} \geq x_{\pi_{2}} \ldots \geq x_{\pi_{p}}$


## Formulation

$$
J_{c}\left(\boldsymbol{y}^{*}, \tilde{\boldsymbol{y}}\right)=\frac{\left|\left\{\boldsymbol{y}^{*}=c\right\} \cap\{\tilde{\boldsymbol{y}}=c\}\right|}{\left|\left\{\boldsymbol{y}^{*}=c\right\} \cup\{\tilde{\boldsymbol{y}}=c\}\right|}
$$

$$
\Delta_{J_{c}}\left(\boldsymbol{y}^{*}, \tilde{\boldsymbol{y}}\right)=1-J_{c}\left(\boldsymbol{y}^{*}, \tilde{\boldsymbol{y}}\right)
$$

$$
f_{i}(c)=\frac{e^{F_{i}(c)}}{\sum_{c^{\prime} \in \mathcal{C}} e^{F_{i}\left(c^{\prime}\right)}} \quad \forall i \in[1, p], \forall c \in \mathcal{C}
$$

$$
m_{i}(c)=\left\{\begin{array}{ll}
{1-f_{i}(c)} & {\text { if } c=y_{i}^{*}} \\
{f_{i}(c)} & {\text { otherwise }}
\end{array}\right.
$$

$$
\operatorname{loss}(\boldsymbol{f})=\frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \overline{\Delta_{J_{c}}}(\boldsymbol{m}(c))
$$

## Results
<div class="fig figcenter fighighlight">
  <img src="/assets/img/lovasz_loss/lovasz_loss_results.png">
</div>

## Implementation

```python
"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard



def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

```

## References
Berman, M., Rannen Triki, A., & Blaschko, M. B. (2018). The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4413-4421).
