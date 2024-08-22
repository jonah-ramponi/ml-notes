---
title: Fancy Pants Attention Techniques
description: Changes to the traditional attention implementation.
date: 2024-03-23
tldr: Many different approaches exist.
draft: false
tags: [attention] 
---

### Sparse Attention
[*Sparse Attention*](https://arxiv.org/pdf/1904.10509v1.pdf) introduces sparse factorizations on the attention matrix. To implement this we introduce a *connectivity pattern* $S = \{S_1,\dots,S_n\}$. Here, $S_i$ denotes the set of indices of the input vectors to which the $i$th output vector attends. For instance, in regular $n^2$ attention every input vector attends to every output vector before it in the sequence. Remember that $d_k$ is the inner dimension of our queries and keys. Sparse Attention is given as follows

\begin{equation*}
    \text{attention}(Q,K,V, S_i) = \text{softmax}\Big( \frac{(Q_{S_i}) K^T_{S_i}}{\sqrt{d_k}} \Big) V_{S_i}.
\end{equation*}

Here, we have defined 

$$ Q_{S_i} = (W_q x_j), K_{S_i} = (W_k x_j), V_{S_i} = (W_v x_j) \text{ for } j \in S_i $$ 

So how do we define the set of connectivity patterns $S$? Formally, we let $S_i = A_i^{h}$ for head $h$ where $A_i^{h} \subset \{j : j \leq i\}$. It is still no clearer how we pick which indices we should take for a given $S_i$. The original authors consider two key criteria initially:


**Criteria 1** 
We should pick $|A_i^h| \propto n^{1/H}$ where $H$ is our total number of heads. This choice is efficient as it ensures the size of the connectivity set scales well with $H$. 

**Criteria 2**
All input positions are connected to output positions across $p$ steps of attention. For instance, for a pair $j \leq i$ we would like $i$ to be able to attend to $j$ through a path of locations with maximum length $p+1$. This helps us propagate signals from input to output in a constant number of steps. 

We now investigate two different approaches that satisfy this criteria, and allow us to implement sparse attention. 

#### Strided Attention. 
We will define a factorized attention pattern in two heads. One head will attend to the previous $l$ locations, while the other head will attend to every $l$th location. We call $l$ the stride and it is chosen to be close to $\sqrt{n}$. 

\begin{align}
    A_i^{(1)} &= \{y,y+1,\dots,i\} \text{ for } t = \max(0,i-l), \\\\
    A_i^{(2)} &= \{j: (i-j)\mod l \equiv 0\}.
\end{align}

Here, $A_i^{(1)}$ simply takes the previous $l$ locations. $A_i^{(2)}$ then takes every $l$th head from the first head where $i-j$ was divisible by $l$ without remainder. This is particularly useful where you can align the structure of your input with the stride. For instance, with a piece of music. Where our input does not have a well defined structured, we use something different. In the image below, you can see $A_i^{(1)}$ responsible for the dark blue shading and $A_i^{(2)}$ responsible for the light blue.

#### Fixed Attention. 
Our goal with this approach is to allow specific cells to summarize the previous locations, and to propagate this information on to future cells.

$$ A^{(1)}_i = \{ j : \text{floor}(\frac{j}{l}) = \text{floor}( \frac{i}{l}) \}, $$
$$ A^{(2)}_i = \{ j : j \mod l \in \{ t, t + 1, \dots, l \} \},  \text{ where } t = l - c \text{ and } c \text{ is a hyperparameter.} $$

These are best understood visually in my opinion. In the image below, $A_i^{(1)}$ is responsible for the dark blue shading and $A_i^{(2)}$ for the light blue shading. If we take stride, $l$ = 128 and $c=8$, then all positions greater than 128 can attend to positions $120-128$. The authors find choosing $c \in \{8,16,32\}$ worked well. 

![Sparse Attention Matrix](/img/sparse_attention.png)

### Sliding Window Attention

[*Sliding Window Attention*](https://arxiv.org/pdf/2004.05150.pdf) reduces the number of calculations we are doing when computing self attention. Previously, to compute attention we took our input matrix of positional encodings $M$, and made copies named $Q, K$ and $V$. We used these copies to compute

\begin{equation}
    \text{attention}(Q,K,V) = \text{softmax}\Big(\frac{Q K^T}{\sqrt{d_k}}\Big) V.
\end{equation}

For now, let's ignore the re-scaling by $\sqrt{d_k}$ and just look at the computation of $QK^T$. This computation looks like
\begin{equation}
    Q \times K^T = \begin{pmatrix}
        Q_{11} & Q_{12} & \cdots & Q_{1d} \\\\
\vdots & \ddots & \cdots & \vdots \\\\
Q_{n1} & Q_{n2} & \cdots & Q_{nd}
\end{pmatrix} \times
\begin{pmatrix}
K_{11} & K_{21} & \cdots & K_{n1} \\\\
\vdots & \ddots & \cdots & \vdots \\\\
K_{1d} & K_{2d} & \cdots & K_{nd}
    \end{pmatrix}
\end{equation}

Our goal is to simplify this computation. Instead of letting each token attend to all of the other tokens, we will define a window size $w$. The token we are calculating attention values for will then only get to look at the tokens $\frac{1}{2}w$ either side of it. For our example, we could consider a sliding window of size $2$ which will look $1$ token to either side of the current token. Only the values shaded in $\colorbox{olive}{olive}$ will be calculated.

![Sliding Window Attention Matrix](/img/sliding_window.png)

This greatly reduces the cost of the computation of $Q \times K^T$, however, the original authors encountered a problem in training. The authors found that this approach is not flexible enough to learn to complete specific tasks. They solved this problem through the introduction of *global attention*. This will give a few of our tokens some special properties: A token with a global attention attends to all other tokens in the sequence and all tokens in the sequence attend to every token with a global attention. 

The local attention (sliding window attention) is primarily used to build contextual representations, while the global attention allows the model to build full sequence representations for prediction. 

We will require two sets of our projection matrices. Firstly, projections to compute attention scores for our sliding window approach $\{Q_s, K_s, V_s\}$ and secondly attention scores for the global attention $\{Q_g,K_g,V_g\}$. These are initialized to the same values.

We first calculate local attention weights using $\{Q_s,K_s,V_s\}$. This gives us an attention output, which is then combined with the output using the global attention weights. The global weights are written on top of the output attention weight matrix calculated by the local attention calculation. 

#### Dilated Sliding Window Attention. 
This is another approach to achieve a similar result. This time, instead of simply taking the $\frac{1}{2}w$ tokens either side of a given $w$ we will introduce some gaps of size $d$. This is referred to as the dilation. Using $w=2, d=1$ in our example we would have an attention matrix which looks like   


![Dilated Sliding Window Attention Matrix](/img/dilated_sliding_window.png)


The authors provide a nice visual of how this looks generally, which you can see in the image below. The authors note they use dilated sliding window attention with small window sizes for lower layers, and larger window sizes for higher layers. They do not introduce dilation for lower layers, however for higher layers a small amount of increasing dilation was introduced on $2$ heads.


![Attention Matrix Visualizations from the Longformer Paper](/img/longformer.png)


### MQA and GQA

#### Multi Query Attention
[*Multi Query Attention*](https://arxiv.org/pdf/1911.02150v1.pdf) (MQA) using the same $K$ and $V$ matrices for each head in our multi head self attention mechanism. For a given head, $h$, $1 \leq h \leq H$, the attention mechanism is calculated as

\begin{equation}
    h_i = \text{attention}(M\cdot W_h^Q, M \cdot W^K,M \cdot W^V).
\end{equation}

For each of our $H$ heads, the only difference in the weight matrices is in $W_h^Q$. Each of these $W_h$ has dimension $(n \times d_q)$. The attention output for each head $i$ is given by  

\begin{equation}
    \text{attention}(Q_h,K,V) = \text{softmax}\Big(\frac{Q_h \cdot K^T}{\sqrt{d_k}} \Big) \cdot V 
\end{equation}

As before, we simply concatenate our attention outputs and multiply by $W^O$, which is defined as before. 

#### Grouped Query Attention
[*Grouped Query Attention*](https://arxiv.org/pdf/2305.13245v3.pdf) (GQA) is very similar to MQA. The difference is that instead of using just one set of $K$, $V$ values for attention calculations it uses $G$ different sets of $K,V$ values. If we have $H$ heads, GQA is equivalent to MHA if $G=H$ and equivalent to MQA if $G=1$. Suppose we want to use $G$ groups. We would firstly allocate each of our $H$ heads into one of the $G$ groups. It would likely make sense to pick $G$ such that $G \mod H \equiv 0$. Though this is not a requirement.

For each  head in a given group, we calculate attention outputs as
\begin{align}
    \text{attention}({h}) &= \text{attention}(M\cdot W_h^Q, M \cdot W^K_g,M \cdot W^V_g) \\\\ 
     &= \text{softmax}\Big(\frac{Q_h \cdot K^T_g}{\sqrt{d_k}} \Big) \cdot V_g 
\end{align}

The query matrices will be shared by all groups under a given head, and the key and value matrices will be used for all attention calculations within a given group. 

#### Conversions from Multi Head Attention. 
A natural question might be how one could take a model which uses multi-head attention and convert it to model using multi query attention or grouped query attention. To convert to multi query attention, we want to find a single representative matrix for both $K$ and $V$ from our set of $H$ different heads. We achieve this via mean pooling. For instance for $K$, 

\begin{equation}
    \text{mean pooling}(K_1,\dots,K_h) \rightarrow K'.
\end{equation}

We need to decide the size of our mean pooling window, $w$. Our process then involves three steps.

Firstly, divide each of the input matrices $(K_1,\dots,K_H)$ into non-overlapping $w \times w$ regions. Then compute the average value within each $w \times w$ region for each input matrix $(K_1,\dots,K_H)$. Finally compute the mean of the corresponding regions across all $H$ input matrices and set this to the corresponding values in our final matrix $K'$.

 We now have our matrix $K'$. It is required at this stage to pre-train for a small portion of the original training steps. The process is nearly identical for grouped query attention. However this time we mean pool over each group of matrices (instead of the whole set). The matrices within a given group are simply dictated by how we chose to assign our $G$ groups to the original $H$ heads. 



### Flash Attention
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


