---
title: Intro to Attention
description: A brief introduction to attention in the transformer architecture. 
date: 2024-03-22
tldr: A brief introduction to attention in the transformer architecture.  
draft: false
tags: [attention, inference] 
---

Suppose you give an LLM the input

*``What is the capital of France?"*

The first thing the LLM will do is split this input into tokens. A token is just some combinations of characters. You can see an example of the tokenization outputs for the question below.

``*$\colorbox{red}{What}\colorbox{magenta}{ is}\colorbox{green}{ the}\colorbox{orange}{ capital}\colorbox{purple}{ of}\colorbox{brown}{ France}\colorbox{cyan}?$*" 

(This tokenization was produced using cl100k_base, the tokenizer used in GPT-3.5-turbo and GPT-4.)

In this example we have $(n = 7)$ tokens. Importantly, from our model's point of view, our input size is defined by the number of tokens instead of words. A numerical representation (vector representation) of each token is now found. Finding this vector representation is called producing an embedding of the token. The token *$\colorbox{red}{ What}$* might get tokenized as follows 

\begin{equation}
    \text{tokenizer}(\textit{\colorbox{red}{What}}) \rightarrow \begin{pmatrix} -0.4159 \\\\  \vdots \\\\   0.5710 \\\\   \end{pmatrix}
\end{equation}

The length of each of our embeddings, these vector outputs of our tokenizer, are the same regardless of the number of characters in our token. Let us denote this length $d_{\text{model}}$. So after we embed each token in our input sequence with our tokenizer we are left with

$$
\begin{pmatrix} -0.415 \\\\  \vdots \\\\   0.571 \\\\   \end{pmatrix} 
\begin{pmatrix} -0.130  \\\\ \vdots  \\\\ 0.192 \\\\ \end{pmatrix}
, \dots ,
\begin{pmatrix} 0.127  \\\\ \vdots \\\\ 0.484 \\\\ \end{pmatrix}
$$


This output is now passed through a *positional encoder*. Broadly, this is useful to provide the model with information about the position of words or tokens within a sequence. You might wonder why we need to positionally encode each token. What does it even mean to positionally encode something? Why can't we just use the index of the item? These questions are for another post.

The only thing that matters for now, is that each of our numerical representations (vectors) are slightly altered. For the numerical representation of the token ``$\colorbox{red}{ What}$" that we get from our embedding model, it might look something like:  

\begin{equation}
     \text{positional encoder}\Bigg(\begin{pmatrix} -0.415 \\\\    \vdots \\\\    0.571 \\\\   \end{pmatrix}\Bigg) = 
    \begin{pmatrix} -0.424 \\\\  \vdots \\\\   0.534 \\\\   \end{pmatrix} 
\end{equation}

Importantly, the positional encoder does not alter the length of our vector, $d_{\text{model}}$. It simply tweaks the values slightly. So far, we entered our prompt: 

> \textit{``What is the capital of Paris?"}

This was tokenized

> ``*$\colorbox{red}{What}\colorbox{magenta}{ is}\colorbox{green}{ the}\colorbox{orange}{ capital}\colorbox{purple}{ of}\colorbox{brown}{ France}\colorbox{cyan}?"$*

Then embedded 

$$
\begin{pmatrix} -0.415 \\\\ -0.514 \\\\  0.569 \\\\  \vdots \\\\  -0.257 \\\\   0.571 \\\\   \end{pmatrix} 
\begin{pmatrix} -0.130 \\\\ -0.464 \\\\ 0.23 \\\\ \vdots \\\\ -0.154 \\\\ 0.192 \\\\ \end{pmatrix}
, \dots ,
\begin{pmatrix} 0.127 \\\\ 0.453 \\\\ 0.110 \\\\ \vdots \\\\ -0.155 \\\\ 0.484 \\\\ \end{pmatrix}
$$

and finally positionally encoded 

\begin{equation}
     \text{positional encoder}\Bigg(\begin{pmatrix} -0.415 \\\\ -0.514 \\\\    \vdots \\\\  -0.257 \\\\   0.571 \\\\   \end{pmatrix}\Bigg) = 
    \begin{pmatrix} -0.424 \\\\ -0.574 \\\\  \vdots \\\\  -0.235 \\\\   0.534 \\\\   \end{pmatrix} 
\end{equation}

We're now very close to being able to introduce attention. One last thing remains, at this point we will transform the output of our positional encoding to a matrix $M$ as follows 

\begin{equation}
    M = \begin{pmatrix}
        -0.424 & -0.574 & 0.513 &  \dots & -0.235 & 0.534 \\\\ 
        -0.133 & 0.461 & 0.228 & \dots & -0.151 & 0.193 \\\\  
        \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\\\
        0.123 & 0.455 & 0.110 & \dots & -0.121 & 0.489
    \end{pmatrix}
    = \text{positional encoding}\begin{pmatrix}
        \text{\colorbox{red}{ What}} \\\\
        \text{\colorbox{magenta}{ is}} \\\\
        \vdots \\\\
        \text{\colorbox{cyan}{?}}
    \end{pmatrix}
\end{equation}

The top row is the first vector output of our positional encoding. The second row is the second, and so on. If we had $n$ tokens in our input sequence, then matrix $M$ would have $n$ rows. The dimensions of $M$ are as follows

\begin{equation}
    M = \Big( \text{number of tokens in input} \times \text{length of embedding}  \Big) = \Big( n \times d_{\text{model}} \Big). 
\end{equation}

### Introduction To Self Attention.
At a high level, self-attention aims to evaluate the importance of each element in a sequence with respect to all other elements and use this to compute a representation of the sequence. All it really does is compute a weighted average of input vectors to produce output vectors. Mathematically, for an input sequence of vectors $x = (x_1, \dots ,x_{n})$ it will return some sequence of vectors, $y = (y_1,\dots,y_m)$ such that 

\begin{equation}
    y_i = \sum_{j = 1}^{{n}} w_{ij} \cdot x_j, \text{ } \forall 1 \leq i \leq m.
\end{equation}

for some mapping $w_{ij}$. The challenge is in figuring out how we should define our mapping $w_{ij}$. Let's look at the first way $w_{ij}$ was defined, introduced in [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf). 

### Scaled Dot Product Self Attention. To compute scaled dot product self attention, we will use the matrix $M$ with rows corresponding to the positionally encoded vectors. $M$ has dimensions $(n \times d_{\text{model}})$. 

We begin by producing query, key and value matrices, analogous to how a search engine maps a user query to relevant items in its database. We will make 3 copies of our matrix $M$. These become the matrices $Q, K$ and $V$. Each of these has dimension $(n \times d_{\text{model}})$. We let $d_k$ denote the dimensions of the keys, which in this case is $d_{\text{model}}$. We are ready to define attention as 

\begin{equation}
    \text{attention}(Q,K,V) = \text{softmax}\Big(\frac{Q K^T}{\sqrt{d_k}}\Big) \cdot V.
\end{equation}

```python
def attention(Q, K, V):
    dk = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(dk)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)
```

Our matrix $QK^T$ of dimension $(n \times d_{\text{model}}) \times (n \times d_{\text{model}})^T = (n \times n)$. After we re-scale by $\sqrt{d_k}$, this matrix is referred to as the *attention matrix*.


**Why do we divide by $\sqrt{d_k}?$** This was introduced to counteract the effect of having the dot products grow large in magnitude for large dimensional inputs $d_k>>1$. In cases where the dot product grew large in size, it was suspect suspected that application of the softmax function was returning extremely small gradients which in turn lead to the vanishing gradients problem.

We multiply the softmax of the attention matrix with each row of $V$. This re-scales each row of the output matrix to sum to one. The equation for softmax applied to a matrix $X$ is as follows

\begin{equation}
    \text{softmax}(X)_{ij} = \frac{e^{X_{ij}}}{\sum_{k=1}^{n} e^{X_{ik}}}.
\end{equation}

```python
def softmax(X):
    exp_X = torch.exp(X)
    denom = exp_X.sum(dim=-1, keepdim=True)
    return exp_X / denom
```

**Why use softmax?** The dot product of $Q$ and $K^T$ gives us a value anywhere between negative and positive infinity. Application of softmax ensures our outputs are more stable. Otherwise, large elements in $Q$ or $K^T$ would grow even larger, dominating the attention mechanism which may cause convergence issues.

Earlier on in we described attention as

\begin{equation}
    y_i = \sum_{j = 1}^{{n}} w_{ij} \cdot x_j, \qquad \forall 1 \leq i \leq m. 
\end{equation}

Well, our *attention matrix* after softmax has been applied is simply $w$ with $(i,j)th$ element $w_{ij}$. The output $y_i$ is just the weighted sum using $w$ on the value vectors, $v = (\vec{v}_1,\dots,\vec{v}_n)$. It may be clearer to visualize the output as

\[
\vec{y} = \begin{pmatrix}
    w_{11} & w_{12} & \dots & w_{1n} \\\\
    w_{21} & w_{22} & \dots & w_{2n} \\\\
    \vdots & \vdots & \ddots & \vdots \\\\
    w_{n1} & w_{n2} & \dots & w_{nn}
\end{pmatrix} \times \begin{pmatrix}
        v_1 \\\\ v_2 \\\\ \vdots \\\\ v_n
    \end{pmatrix}
\]

The attention matrix is a nice thing to visualize. For our toy example, it might look like 


<att-m>

What can we notice about our attention matrix?

**It is symmetric.**
That is, $w = w^T$. This is to be expected, as remember it was produced by computing $QK^T$ where $Q$ and $K$ are identical.

**The largest values are often times found on the leading diagonal.** 
You can think of the values in the matrix as some measure of how important one token is to another. Typically, we try to ensure that each token pays attention to itself to some extent. 

**Every cell is filled.**
This is because in this attention approach, every token attends to every other token. This is often referred to as *full $n^2$ attention*. 

#### Multi Head Self Attention.

It's important to acknowledge that there may not exist a single perfect representation of the attention matrix. Multi Head Self Attention allows us to produce many different representations of the attention matrix. Each individual attention mechanism is referred to as a ``head". Each head learns slightly different representations of the input sequence, which the original researchers found prompted the best output. Firstly, we're going to introduce some new matrices. These will be defined as

\begin{align*}
    Q = (n \times d_q), \hspace{3mm} K = (n \times d_k), \hspace{3mm} V = (n \times d_v)
\end{align*}

These matrices will be obtained by linearly transforming the original matrix $M$, using weight matrices $W^Q$, $W^K$ and $W^V$ respectively: 
\begin{align*}
    Q &= M\times W^Q, \\\\
    K &= M \times W^K, \\\\
    V &= M \times W^V.
\end{align*}

Each of these matrices has $d_{\text{model}}$ rows, and remember that $M$ has $d_{\text{model}}$ columns. We have control over parameters $d_q, d_k, d_v$. In the original research they took $d_q = d_k =  d_v = d_{\text{model}}/8 = 64$.

We're going to use a different set of weight matrices $W^Q$, $W^K$ and $W^V$ for each head. If we have $H$ heads, we will refer to the set of weight matrices of the $h_{th}$ head as $\{ W_h^Q, W_h^K, W_h^V \}$. For a given head, $h$, the output of the attention mechanism is

\begin{equation}
    h_i = \text{attention}(M \cdot W_h^Q, M\cdot W_h^K,M\cdot W_h^V)
\end{equation}

The overall output of the process is then simply 

\begin{equation}
       \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \cdots, \text{head}_H)W^O.  
\end{equation}

Concat() simply concatenates our output matrices. The output matrix of size $(n \times d_v)$ for each head is simply our matrices stacked on top of one another like so

\begin{equation*}
    \text{Concat}(\text{head}_1, \dots, \text{head}_h) = 
    \begin{pmatrix}
        \text{head}_{1_{11}} & \dots & \text{head}_{1_{1d_v}} & \dots & \text{head}_{H_{11}} & \dots & \text{head}_{H_{1d_v}} \\\\ 
        \text{head}_{1_{21}} & \dots & \text{head}_{1_{2d_v}} & \dots & \text{head}_{H_{21}} & \dots & \text{head}_{H_{2d_v}} \\\\ 
        \vdots & \ddots & \vdots & \dots & \vdots & \ddots & \vdots \\\\
        \text{head}_{1_{n1}} & \dots & \text{head}_{1_{nd_v}} & \dots & \text{head}_{H_{n1}} & \dots & \text{head}_{H_{nd_v}} \\\\ 
    \end{pmatrix}
\end{equation*}

This output has dimension $(n \times H d_v)$. We still have $n$ rows, however now we have $h$ different representations of $d_v$. Our output, $W^O$, is another trainable weight matrix which has dimensions $W^O = (Hd_v \times d_{\text{model}})$. Therefore, the multiplication of Concat $(\text{head}_1, \dots, \text{head}_H)$ and $W^O$ results in a matrix with dimension $(n \times d_{\text{model}})$.