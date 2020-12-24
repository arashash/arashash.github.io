---
layout: post
title: Boundary Loss for Image Segmentation
date: '2020-01-06 21:50'
excerpt: This blog post will guide you on how to understand, implement, and tune a boundary loss function for image segmentation in Pytorch.
comments: true
---

## Formulation

\begin{equation}
y_{g t}^{b}=\operatorname{pool}\left(1-y_{g t}, \theta_{0}\right)-\left(1-y_{g t}\right), y_{p d}^{b}=\operatorname{pool}\left(1-y_{p d}, \theta_{0}\right)-\left(1-y_{p d}\right)
\end{equation}

\begin{equation}
y_{g t}^{b, e x t}=\operatorname{pool}\left(y_{g t}^{b}, \theta\right), y_{p d}^{b, e x t}=\operatorname{pool}\left(y_{p d}^{b}, \theta\right)
\end{equation}

\begin{equation}
P^{c}=\frac{\operatorname{sum}\left(y_{p d}^{b} \circ y_{g t}^{b, e x t}\right)}{\operatorname{sum}\left(y_{p d}^{b}\right)}, R^{c}=\frac{\operatorname{sum}\left(y_{g t}^{b} \circ y_{p d}^{b, e x t}\right)}{\operatorname{sum}\left(y_{g t}^{b}\right)}
\end{equation}

\begin{equation}
B F_{1}^{c}=\frac{2 P^{c} R^{c}}{P^{c}+R^{c}}, L_{B F_{1}^{c}}=1-B F_{1}^{c}
\end{equation}

## Hyperparameters
Acccording to the paper [1], "Hyperparameter  $\theta_0$ must be as small as possible to extract vicious boundary; usually we set $\theta_0=3$".

And, "The value of hyperparameter $\theta$ can be determined as not greater than the minimum distance between neighboring segments of the binary ground truth map".

## Results
<div class="fig figcenter fighighlight">
  <img src="/assets/img/boundary-loss/boundary_loss.png">
  <div class="figcaption"><br>The loss calculation of a class with $\theta=\theta_0=9$; (b) ground truth segment (gt); (c) predicted segment
(pred); (d) boundary of gt; (e) boundary of pred; (f) expanded boundary of gt; (g)
expanded boundary of pred; (h) pixel-wise multiplication of masks (d) and (g); (i)
pixel-wise multiplication of masks (e) and (f)<br>
  </div>
</div>

The boundary F1 score of the same label and prediction in the figure with different combinations of the hyperparameters:

|               | $\theta=3$ | $\theta=5$ | $\theta=7$ | $\theta=9$ | $\theta=11$ | $\theta=13$ | $\theta=15$ | $\theta=17$ | $\theta=19$ |
|---------------|------------|------------|------------|------------|-------------|-------------|-------------|-------------|-------------|
| $\theta_0=3$  | 0.06       | 0.07       | 0.12       | 0.20       | 0.29        | 0.47        | 0.60        | 0.80        | 0.88        |
| $\theta_0=5$  | 0.06       | 0.09       | 0.16       | 0.24       | 0.38        | 0.53        | 0.70        | 0.84        | 0.91        |
| $\theta_0=7$  | 0.08       | 0.12       | 0.20       | 0.32       | 0.45        | 0.62        | 0.76        | 0.87        | 0.94        |
| $\theta_0=9$  | 0.10       | 0.16       | 0.27       | 0.38       | 0.54        | 0.68        | 0.81        | 0.92        | 0.97        |
| $\theta_0=11$ | 0.14       | 0.22       | 0.33       | 0.46       | 0.61        | 0.76        | 0.88        | 0.96        | 0.98        |
| $\theta_0=13$ | 0.19       | 0.28       | 0.41       | 0.55       | 0.70        | 0.83        | 0.92        | 0.96        | 0.98        |
| $\theta_0=15$ | 0.25       | 0.37       | 0.50       | 0.65       | 0.79        | 0.89        | 0.94        | 0.97        | 0.98        |
| $\theta_0=17$ | 0.34       | 0.47       | 0.60       | 0.74       | 0.85        | 0.91        | 0.94        | 0.97        | 0.98        |
| $\theta_0=19$ | 0.44       | 0.57       | 0.71       | 0.82       | 0.88        | 0.92        | 0.95        | 0.97        | 0.98        |

### Bad Example
The IoU score on this example is 0.18:

Heatmap of F1 scores same as the table             |  Ground Truth and Prediction
:-------------------------:|:-------------------------:
![](/assets/img/boundary-loss/heatmap_IoU_0.18.png)  |  ![](/assets/img/boundary-loss/imgs_IoU_0.18.jpg)

### Medium Example
The IoU score on this example is 0.56:

Heatmap of F1 scores same as the table             |  Ground Truth and Prediction
:-------------------------:|:-------------------------:
![](/assets/img/rnn/boundary-loss/heatmap_IoU_0.56.png)  |  ![](/assets/img/rnn/boundary-loss/imgs_IoU_0.56.jpg)

### Good Example
The IoU score on this example is 0.74:

Heatmap of F1 scores same as the table             |  Ground Truth and Prediction
:-------------------------:|:-------------------------:
![](/assets/img/rnn/boundary-loss/heatmap_IoU_0.74.png)  |  ![](/assets/img/rnn/boundary-loss/imgs_IoU_74.jpg)


## Implementation

```python
class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=5, theta=11):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """      
        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        eps = 1e-7
        P = (torch.sum(pred_b * gt_b_ext, dim=2) + eps) / (torch.sum(pred_b, dim=2) + eps)
        R = (torch.sum(pred_b_ext * gt_b, dim=2) + eps) / (torch.sum(gt_b, dim=2) + eps)

        # Boundary F1 Score
        BF1 = (2 * P * R + eps) / (P + R + eps)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1-BF1)


        return loss
```

## References
[1] Bokhovkin, A., & Burnaev, E. (2019, July). Boundary Loss for Remote Sensing Imagery Semantic Segmentation. In International Symposium on Neural Networks (pp. 388-401). Springer, Cham.
