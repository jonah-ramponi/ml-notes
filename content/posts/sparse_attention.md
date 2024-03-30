---
title: Sparse Attention
description: Reducing the number of calculations to compute attention.
date: 2024-03-22
tldr: Reducing the number of calculations to compute attention.
draft: false
tags: [attention, inference] 
---

\textit{Sparse Attention} \cite{sparseatt} introduces sparse factorizations on the attention matrix. To implement this we introduce a \textit{connectivity pattern} $S = \{S_1,\dots,S_n\}$. Here, $S_i$ denotes the set of indices of the input vectors to which the $i$th output vector attends. For instance, in regular $n^2$ attention every input vector attends to every output vector before it in the sequence. Remember that $d_k$ is the inner dimension of our queries and keys. Sparse Attention is given as follows

\begin{align}
    \text{attention}(Q,K,V, S_i) &= \text{softmax}\Big( \frac{(Q_{S_i}) K^T_{S_i}}{\sqrt{d_k}} \Big) V_{S_i}.
\end{align}

Here, we have defined

\begin{align*}
    Q_{S_i} &= \big(W_q \vec{x}_j \big)_{j \in S_i}, \\\\
    K_{S_i} &= \big(W_k \vec{x}_j \big)_{j \in S_i}, \\\\
    V_{S_i} &= \big(W_v\vec{x}_j \big)_{j \in S_i}. 
\end{align*}

So how do we define the set of connectivity patterns $S$? Formally, we let $S_i = A_i^{h}$ for head $h$ where $A_i^{h} \subset \{j : j \leq i\}$. It is still no clearer how we pick which indices we should take for a given $S_i$. The original authors consider two key criteria initially:


#### Criteria 1 
We should pick $|A_i^h| \propto n^{1/H}$ where $H$ is our total number of heads. This choice is efficient as it ensures the size of the connectivity set scales well with $H$. 

#### Criteria 2
All input positions are connected to output positions across $p$ steps of attention. For instance, for a pair $j \leq i$ we would like $i$ to be able to attend to $j$ through a path of locations with maximum length $p+1$. This helps us propagate signals from input to output in a constant number of steps. 

We now investigate two different approaches that satisfy this criteria, and allow us to implement sparse attention. 

**Strided Attention.** We will define a factorized attention pattern in two heads. One head will attend to the previous $l$ locations, while the other head will attend to every $l$th location. We call $l$ the stride and it is chosen to be close to $\sqrt{n}$. 

\begin{align}
    A_i^{(1)} &= \{y,y+1,\dots,i\} \text{ for } t = \max(0,i-l), \\\\
    A_i^{(2)} &= \{j: (i-j)\mod l \equiv 0\}.
\end{align}

Here, $A_i^{(1)}$ simply takes the previous $l$ locations. $A_i^{(2)}$ then takes every $l$th head from the first head where $i-j$ was divisible by $l$ without remainder. This is particularly useful where you can align the structure of your input with the stride. For instance, with a piece of music. Where our input does not have a well defined structured, we use something different. In the image below, you can see $A_i^{(1)}$ responsible for the dark blue shading and $A_i^{(2)}$ responsible for the light blue.

**Fixed Attention*.* Our goal with this approach is to allow specific cells to summarize the previous locations, and to propagate this information on to future cells.

\begin{align*}
    A^{(1)}_i &= \Big\{ j : \left\lfloor \frac{j}{l} \right\rfloor = \left\lfloor \frac{i}{l} \right\rfloor \Big\}, \\\\
    A^{(2)}_i &= \Big\{ j : j \mod l \in \{ t, t + 1, \ldots, l \} \Big\},  \text{ where } t = l - c \text{ and } c \text{ is a hyperparameter.}
\end{align*}

These are best understood visually in my opinion. In the image below, $A_i^{(1)}$ is responsible for the dark blue shading and $A_i^{(2)}$ for the light blue shading. If we take stride, $l$ = 128 and $c=8$, then all positions greater than 128 can attend to positions $120-128$. The authors find choosing $c \in \{8,16,32\}$ worked well. 

![my alt text](/img/sparse_attention.png)