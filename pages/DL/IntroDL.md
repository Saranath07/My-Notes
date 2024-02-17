---
title: Backpropogation
---

Backpropogation is a technique which is used to find the gradients of each layer with respect to the loss. We can write our loss function as:

$$
\hat{y}_i = \hat{f}(x_i) = O(W_3 \ g(W_2 \ g(W_1 x_i + b_1) + b_2) + b_3)
$$

This is when we have 3 inputs, 2 hidden layers with 3 neurons in each and 1 output layer.

![Example Feed Forward Neural Network](/images/Neuron-1.png)

We use the optimization algorithm **Gradient Descent** which can be used to traverse the loss function and get into the local minima.

The update rule goes like this:

$$
\theta_{t+1} = \theta_{t} - \eta \nabla_{\theta}\mathcal{L}(\theta)
$$

where
$$
\begin{align*}
\theta  &  :     \text{Parameters} \\
\eta & :  \text{Learning Rate} \\
\nabla_{\theta}\mathcal{L}(\theta) & : \text{Gradient of loss function with respect to the parameters}.
\end{align*}
$$

As we have more parameters and functions, computation of gradient with respect to the parameters i.e. $\nabla_{\theta}\mathcal{L}(\theta)$ is not straight forward.

So we depend on chain rule to get the solution.

Let us try to compute just the loss with respect to $1$ parameter.

$$
\dfrac{\partial \mathcal{L}(\theta)}{\partial W_{111}} = \dfrac{\partial \mathcal{L}(\theta)}{\partial \hat{y}}\dfrac{\partial \hat{y}}{\partial a_{L11}}\dfrac{\partial a_{L11}}{\partial h_{21}}\dfrac{\partial h_{21}}{\partial a_{21}}\dfrac{\partial a_{21}}{\partial h_{11}}\dfrac{\partial h_{11}}{\partial a_{11}}\dfrac{\partial a_{11}}{\partial W_{111}}
$$
