---
title: Flash Attention
description: Reduce the memory usage used to compute exact attention.
date: 2024-03-26
tldr: Reduce the memory usage used to compute exact attention.
draft: false
tags: [attention, inference] 
---
The goal of [*Flash Attention*](https://arxiv.org/pdf/2205.14135.pdf) is to compute the attention value with fewer high bandwidth memory read / writes. The approach has since been refined in  [*Flash Attention 2*](https://arxiv.org/pdf/2307.08691.pdf). 

We will split the attention inputs $Q,K,V$ into blocks. Each block will be handled separately, and attention will therefore be computed with respect to each block. With the correct scaling, adding the outputs from each block we will give us the same attention value as we would get by computing everything all together. 

**Tilling.** To compute attention, we multiply $Q \times K^T$, divide by $\sqrt{d_k}$ and then take the softmax. Keeping track of the scaling values in softmax is the key to making this technique work. The softmax for a vector $\vec{x} \in \mathbb{R}^{2n}$ is given by

$$
    m(x):= \max_i x_i, \hspace{3mm} f(x):= [e^{x_1-m(x)}, \dots, e^{x_b -m(x)}], \hspace{3mm} \ell(x) := \sum_i f(x)_i, \hspace{3mm} \text{softmax}(x) := \frac{f(x)}{\ell(x)}.
$$

This looks unfriendly, but is really just the notation for a more numerically stable softmax. What does that mean? Well, notice we are just applying regular softmax but with some shifting of each element of vector $\vec{x}$ by $\max(x)$ units. We can do this because softmax$(\vec{x}) = \text{softmax}(\vec{x}-c)$ for any scalar $c$. 


*Proof* 
\begin{align*}
\text{softmax}(\vec{x} - c) &= \frac{e^{\vec{x} - c}}{\sum_{j} e^{x_j - c}} \\\\
&= \frac{e^{\vec{x}} \cdot e^{-c}}{\sum_{j} e^{x_j} \cdot e^{-c}} \\\\
&= \frac{e^{\vec{x}}}{\sum_{j} e^{x_j}} \\\\
&= \text{softmax}(\vec{x})
\end{align*}

In this case, we improve numerical stability by ensuring we do not take the exponential of very large numbers. This can lead to overflow issues. This simply means our number gets too big to store in the given datatype. By subtracting the largest element, we ensure the vector $\vec{x}$ only has non-positive entries. For example, in floating point 64, the maximum value we can represent is very large $(10^{308})$. However

$$
    e^x > 10^{308} \implies x > \ln(10^{308}) \implies x > 308 \times \ln(10) \implies x > 709.
$$

Therefore, approximately any $x$ larger than $709$ will result in overflow issues. For instance, computing $\exp(709) = 8.22e+307$ but $\exp(710) = inf$ in *numpy*. 

```python
np.exp(709)
# 8.218407461554972e+307
```

```python
np.exp(710)
# <stdin>:1: RuntimeWarning: overflow encountered in exp
# inf
```

We certainly do not want our model to hit any overflow errors. It is therefore preferable to use this numerically stable version of softmax. 


To compute softmax in blocks, we decompose our vector $\vec{x} \in \mathbb{R}^{2n}$ into two smaller vectors in $\mathbb{R}^n$.Let's look at the simple case of decomposing into two vectors. Denote these vectors $\vec{x}_1,\vec{x}_2$ each in $\mathbb{R}^n$. Our softmax calculation becomes

\begin{aligned}
    m(x) &= m([x_1\hspace{3mm}  x_2]) = \max (m(x_1),m(x_2)), \\\\
    f(x) &= [e^{m(x_1) - m(x)}f(x_1) \hspace{3mm} e^{m(x_2) - m(x)}f(x_2)], \\\\
    \ell(x) &= \ell([x_1\hspace{3mm}  x_2]) = [e^{m(x_1) - m(x)}\ell(x_1) \hspace{3mm} e^{m(x_2) - m(x)}\ell(x_2)], \\\\
    \text{softmax}(x) &= \frac{f(x)}{\ell(x)}.
\end{aligned}

Notice that we use $m(x_i) - m(x)$ as the normalization factor, as we do not know which group will contain the maximum value of $\vec{x}$. By keeping track of both $m(x)$ and $\ell(x)$ we will be able to accurately recombine the softmax outputs for each block, as will know how to rescale the softmax outputs.

**Recomputation.** We also do not wish to store all the intermediate values we calculate for every backward pass. Typically we require the attention matrix, $QK^T$, and the output after softmax, simply softmax($QK^T$) in each backward pass. However, by using our blocks of $Q,K,V$ the whole attention matrix is not required to be loaded in during every backward pass. 


