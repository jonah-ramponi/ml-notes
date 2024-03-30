---
title: Sliding Window Attention
description: Altering the tokens to which a token in the input sequence attends.
date: 2024-03-22
tldr: Altering the tokens to which a token in the input sequence attends.
draft: false
tags: [attention, inference] 
---

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

Our goal is to simplify this computation. Instead of letting each token attend to all of the other tokens, we will define a window size $w$. The token we are calculating attention values for will then only get to look at the tokens $\frac{1}{2}w$ either side of it. For our example, we could consider a sliding window of size $2$ which will look $1$ token to either side of the current token. Only the values shaded in \colorbox{olive}{olive} will be calculated.

![Sliding Window Attention Matrix](/img/sliding_window.png)

This greatly reduces the cost of the computation of $Q \times K^T$, however, the original authors encountered a problem in training. The authors found that this approach is not flexible enough to learn to complete specific tasks. They solved this problem through the introduction of *global attention*. This will give a few of our tokens some special properties: A token with a global attention attends to all other tokens in the sequence and all tokens in the sequence attend to every token with a global attention. 

The local attention (sliding window attention) is primarily used to build contextual representations, while the global attention allows the model to build full sequence representations for prediction. 

We will require two sets of our projection matrices. Firstly, projections to compute attention scores for our sliding window approach $\{Q_s, K_s, V_s\}$ and secondly attention scores for the global attention $\{Q_g,K_g,V_g\}$. These are initialized to the same values.

We first calculate local attention weights using $\{Q_s,K_s,V_s\}$. This gives us an attention output, which is then combined with the output using the global attention weights. The global weights are written on top of the output attention weight matrix calculated by the local attention calculation. 

**Dilated Sliding Window Attention.** is another approach to achieve a similar result. This time, instead of simply taking the $\frac{1}{2}w$ tokens either side of a given $w$ we will introduce some gaps of size $d$. This is referred to as the dilation. Using $w=2, d=1$ in our example we would have an attention matrix which looks like   


![Dilated Sliding Window Attention Matrix](/img/dilated_sliding_window.png)


The authors provide a nice visual of how this looks generally, which you can see in the image below. The authors note they use dilated sliding window attention with small window sizes for lower layers, and larger window sizes for higher layers. They do not introduce dilation for lower layers, however for higher layers a small amount of increasing dilation was introduced on $2$ heads.


![Attention Matrix Visualizations from the Longformer Paper](/img/longformer.png)
