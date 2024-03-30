---
title: "Post 2"
date: 2024-03-30T11:49:13Z
draft: false
---

Here's my second content [View as PDF](/posts/file/Attention_Mechanisms.pdf). 

> Here is a quote by me 

![my alt text](/img/longformer.png)


```python
def attention(Q, K, V):
    dk = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(dk)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)
```

{{< youtube id="q389nbmv4MU" >}}

test 2 
