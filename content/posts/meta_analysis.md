---
title: Why did Meta publish the Llama models for free?
description: My thoughts on Meta's strategic decision to open source the llama model weights. 
date: 2024-08-23
tldr: Meta makes money from adverts
draft: false
tags: [] 
---

To understand why Meta has open sourced the Llama family of models, it is important to understand how Meta makes money. Meta makes money from adverts. Almost their entire revenue comes from adverts (1). So, 

why have Meta invested so much money into the Llama models?

*Probably, to make more money from adverts.*

Here are some ways in which the open source release of the Llama models might help Meta make more money from adverts. 

### To increase the value of their data

Open sourcing the Llama model weights has led to massive adoption of the Llama models by developers and researchers. This community-driven effort has greatly advanced understanding of generative AI models. These advancements should enable Meta to extract more value from their data.

A while back, a leaked memo from a Google Researcher titled *“We Have No Moat, And Neither Does OpenAI“* was released (2). Meta promptly decided to focus on building bridges with the open source community. They do, however, keep a deep moat filled with crocodiles around what is truly valuable to them - their data.

It would not surprise me if we never hear about the models trained on our Instagram and Facebook data. I think it will be these models which will reap the biggest rewards for Meta - because those models might have the potential to increase advertising revenues by a percentage or two, and that is worth a lot to Meta.

So what exactly might this value add look like?

I don’t know. That’s a question for the researchers at Meta under super strict NDAs. But maybe:  

1. Improvements to multimodal models which could allow them to extract more information from user’s posts. For instance, they could better identify the types, values, styles, etc., of the clothes an individual wears. They might better understand the look they go for. This would enhance targeted advertising. 

2. Improvements to language models may allow for better analysis of the tone and sentiment in posts. Meta could then tailor ads to match the user’s current emotional state, improving engagement and relevance. For instance, if emotional stress were detected Meta could recommend *using Facebook less*. 

But why didn’t Meta just train the Llama models privately and keep them for internal use? Why expose themselves to potential legal issues or lawsuits by making them open source?

I believe the prevailing theory is that Meta is strategically commoditizing its complements (3). They have a core revenue stream—advertising—and by making certain technologies widely available, they indirectly enhance their primary money-making activity. 

I would guess that Meta intends to commoditize as much of their tech stack as possible. For instance, by creating a new frontend framework which makes it easier to build scalable social network sites. Or by creating a better framework than tensorflow for building digital advertising ML models. 

Meta can initial development this to serve what they need internally, providing top quality engineering for the initial codebase. Once released to the open-source community, developers around the world contribute to refining and expanding its functionality.

When a company like Meta finds itself behind the competition in certain areas, open-sourcing its work will help bridge the gap. This approach ensures that Meta catches up thanks to continual improvements by the community. These developments feed back into Meta’s ultimate goal—better advertising revenues

### To recruit
Open-sourcing the Llama models exposes thousands of developers to Meta’s technology. New hires, particularly researchers, may already be familiar with Meta’s tools and models, reducing the need for extensive training. 

Furthermore, by publishing models for free, Meta will attract researchers and developers who want to work on cutting-edge AI technologies. It might not be quite as good an advert as *The Internship* was for Google, but I believe that the decision to open source the Llama models has improved Meta’s reputation amongst much of the community. And that is worth something.

### Because big tech is a battlefield
In 1812, during Napoleon Bonaparte’s invasion of Russia, the Russian army employed a ruthless strategy as they retreated: they burned crops, villages, and resources to deny the advancing French any supplies. This *"Scorched Earth"* tactic is a military strategy where anything that could aid the enemy is destroyed.

With Generative AI, Meta is doing the opposite. Let’s compare AI to whoopee cushions.

Imagine a world in which one company has a strategy to produce the best whoopee cushions (Microsoft). Initially, they have seemingly total dominance. Nobody is safe. 

To counter Microsoft’s dominance, Meta takes whoopee cushions and gives them to everyone. They teach everyone how to make them. Microsoft is now far less big and scary to Meta. Their threats of price increases on the Big Whooper 3000 model can no longer be justified. The power is no longer concentrated in Microsoft's hands as everyone now has open access to whoopee cushions.

However, this strategy isn't as simple as it seems. While it might appear that Meta is levelling the playing field, not everyone can leverage these whoopee cushions effectively. Small players should benefit, but large enterprises might still prefer Microsoft's integrated solutions, even at a higher cost. Microsoft's dominance isn't just about the whoopee cushions themselves; it's about the entire flatulent ecosystem they control.

Whether the world is a better place in this example is up for debate. 

I’m sure many people would rather live in a world with no whoopee cushions at all.

Big tech is full of battlefield-like tactics. Another example is Alphabet’s dominance in the smartphone market. For a while Facebook considered developing phones. To make these phones, Facebook wanted their own operating system and their own devices, and needed navigation software for it. So, So, Facebook tried to buy Waze (a similar service to Google Maps).  

There were many other reasons Facebook wanted Waze; one key reason being to deliver better location-based mobile advertising and optimise local search results. However, right before they could buy Waze for 500 million dollars, Google stepped in and paid nearly a billion dollars to snatch it up. 

Why? I think that Google probably bought Waze so that Facebook would not have it. They probably didn't really need Waze, but it contributed to Facebook abandoning their smartphone plans. It stunted Facebook's expansion. For big tech, messing up each other's game is a key, legitimate business strategy.

Similarly, Microsoft's (and OpenAI’s) dominance in Generative AI may have been enough of a threat to prompt Meta into action. By making AI tools freely available, Meta isn't simply playing nice—it’s playing smart. It’s a calculated move to dilute Microsoft’s power and shift the balance in the game, ensuring that no one company holds too much sway in the AI landscape.

So, while Meta’s actions might seem altruistic on the surface, they are driven by the same competitive instincts that have shaped big tech for years. 

Whether this leads to a better world, with more innovation and access, or just a noisier one full of whoopee cushions, remains to be seen.

(1) In the second quarter of 2024, 98.1% of Meta’s revenue came from adverts. [Meta Q2 Earnings](ps://investor.fb.com/investor-news/press-release-details/2024/Meta-Reports-Second-Quarter-2024-Results/default.aspx)

(2) [We Have No Moat, And Neither Does OpenAI](https://www.semianalysis.com/p/google-we-have-no-moat-and-neither)

(3) [Joel on Software discusses Commoditizing your Complement](https://www.joelonsoftware.com/2002/06/12/strategy-letter-v/)
