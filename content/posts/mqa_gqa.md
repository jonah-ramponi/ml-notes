---
title: Multi & Grouped Query Attention
description: Use less K and V matrices to use less memory.
date: 2024-03-22
tldr: Use less K and V matrices to use less memory.
draft: false
tags: [attention, inference] 
---

[*Multi Query Attention*](https://arxiv.org/pdf/1911.02150v1.pdf) (MQA) using the same $K$ and $V$ matrices for each head in our multi head self attention mechanism. For a given head, $h$, $1 \leq h \leq H$, the attention mechanism is calculated as

\begin{equation}
    h_i = \text{attention}(M\cdot W_h^Q, M \cdot W^K,M \cdot W^V).
\end{equation}

For each of our $H$ heads, the only difference in the weight matrices is in $W_h^Q$. Each of these $W_h$ has dimension $(n \times d_q)$. The attention output for each head $i$ is given by  

\begin{equation}
    \text{attention}(Q_h,K,V) = \text{softmax}\Big(\frac{Q_h \cdot K^T}{\sqrt{d_k}} \Big) \cdot V 
\end{equation}

As before, we simply concatenate our attention outputs and multiply by $W^O$, which is defined as before. 


[*Grouped Query Attention*](https://arxiv.org/pdf/2305.13245v3.pdf) (GQA) is very similar to MQA. The difference is that instead of using just one set of $K$, $V$ values for attention calculations it uses $G$ different sets of $K,V$ values. If we have $H$ heads, GQA is equivalent to MHA if $G=H$ and equivalent to MQA if $G=1$. Suppose we want to use $G$ groups. We would firstly allocate each of our $H$ heads into one of the $G$ groups. It would likely make sense to pick $G$ such that $G \mod H \equiv 0$. Though this is not a requirement.

For each  head in a given group, we calculate attention outputs as
\begin{align}
    \text{attention}({h}) &= \text{attention}(M\cdot W_h^Q, M \cdot W^K_g,M \cdot W^V_g) \\\\ 
     &= \text{softmax}\Big(\frac{Q_h \cdot K^T_g}{\sqrt{d_k}} \Big) \cdot V_g 
\end{align}

The query matrices will be shared by all groups under a given head, and the key and value matrices will be used for all attention calculations within a given group. 

**Conversions from Multi Head Attention.** A natural question might be how one could take a model which uses multi-head attention and convert it to model using multi query attention or grouped query attention. To convert to multi query attention, we want to find a single representative matrix for both $K$ and $V$ from our set of $H$ different heads. We achieve this via mean pooling. For instance for $K$, 

\begin{equation}
    \text{mean pooling}(K_1,\dots,K_h) \rightarrow K'.
\end{equation}

We need to decide the size of our mean pooling window, $w$. Our process then involves three steps.

Firstly, divide each of the input matrices $(K_1,\dots,K_H)$ into non-overlapping $w \times w$ regions. Then compute the average value within each $w \times w$ region for each input matrix $(K_1,\dots,K_H)$. Finally compute the mean of the corresponding regions across all $H$ input matrices and set this to the corresponding values in our final matrix $K'$.

 We now have our matrix $K'$. It is required at this stage to pre-train for a small portion of the original training steps. The process is nearly identical for grouped query attention. However this time we mean pool over each group of matrices (instead of the whole set). The matrices within a given group are simply dictated by how we chose to assign our $G$ groups to the original $H$ heads. 
