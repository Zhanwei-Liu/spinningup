===================================
深度强化学习研究者的资料
===================================
By Joshua Achiam, October 13th, 2018


.. contents:: Table of Contents
    :depth: 2

如果你是一位深度强化学习的研究者，你现在可能已经对深度强化学习有了很多的了解。你知道它 `很难`_ 而且 `不总是有效`_ 。即便是严格按照步骤来， `可重现性`_ 依然是一大挑战。如果你准备从头开始， `学习的曲线非常陡峭`_ 。虽然已经有很多很棒的学习资源（比如 ），但是因为很多资料都很新，以至于还没有一条清晰明确的捷径。这个项目的目的就是帮助你克服这些一开始的障碍，并且让你清楚的知道，如何成为一名深度强化学习研究院。在这个项目里，我们会介绍一些有用的课程，作为基础知识，同时把一些可能适合研究的方向结合进来。

正确的背景
====================

**建立良好的数学背景** 从概率和统计学的角度，要对于随机变量、贝叶斯定理、链式法则、期望、标准差和重要性抽样等要有很好的理解。从多重积分的角度，要了解梯度和泰勒展开（可选，但是会很有用）。

**对于深度学习要有基础的了解** 你不用知道每一个技巧和结构，但是了解基础的知识很有帮助。要了解多层感知机额、LSTM、GRU、卷积、层、resnets、注意力机制、mechanisms，常见的正则手段(`weight decay`_, `dropout`_)，归一化方式(`batch norm`_, `layer norm`_, `weight norm`_)和优化方式(`SGD, momentum SGD`_, `Adam`_, `以及其它`_)。要了解什么是  `reparameterization trick`_ 。

**至少熟悉一种深度学习框架** `Tensorflow`_ or `PyTorch`_ 非常适合练手。你不用知道所有东西，但是你要能非常自信的实现一种监督学习算法。

**对于强化学习中的主要概念和术语很了解** 知道什么是状态、行动、轨迹、策略、奖励、值函数和行动值函数。如果你对这些不了解，去读一读项目里面 `介绍`_ 部分的材料。OpenAI Hackthon 的 `强化学习介绍`_ 也很值得看，或者是 Lilian Weng 的 综述。如果你对于数学理论很感兴趣，可以学习 `monotonic improvement theory`_ （策略梯度算法的的基础）或者 `classical RL algorithms`_ （尽管被深度强化学习所替代，但还是有很多能推动新的研究的 insights）。

在动手中学习
==============

**自己实现算法** 你应该尽可能地从头开始编写尽可能多的深度学习的核心算法，同时要保证自己的实现尽量简单、正确。这是了解这些算法如何工作、培养性能特征直觉的最佳方法。

**简单是最重要的** 你要对自己的工作有合理的规划，从最简单的算法开始，然后慢慢引入复杂性。如果你一开始就构建很多复杂的部分，有可能会耗费你接下来几周的时间来尝试调试。对于刚刚接触强化学习的人来说，这是很常见的问题。如果你发现自己被困在其中，不要气馁，尝试回到最开始然后换一种更简单的算法。

**哪些算法？** 你可以按照 vanilla policy gradient(也被称为 `REINFORCE`_ )、`DQN`_, `A2C`_ ( `A3C`_ 的同步版本), `PPO`_ (具有 clipped objective 特性的变体), `DDPG` 的顺序来学习。 这些算法的最简版本可以用几百行代码编写（大约250-300行），有些更少，比如 `a no-frills version of VPG`_ 只需要 80 行的代码。再写并行版本代码之前，先尝试写单线程版本的。（至少实现一种并行的算法）

**注重理解** 编写有效的强化学习代码需要对于算法有明确的理解，同时注重细节。因为错误的代码总是悄无声息：看起来运行的很正常，但实际上智能体什么也没有学到。这种情况通常是因为有些公式写错了，或者分布不对，又或者数据传输到了错误的地方。有时候找到这些错误的唯一办法，就是批判性地阅读代码，明确知道它应该做什么，找到它偏离正确行为的地方。这就需要你一方面了解学术文献，另一方面参考已有的实现，所以你要花很多时间在这些上面。

**看论文的时候要注意什么** 当基于一篇论文实现算法的时候，要彻读论文

**But don't overfit to paper details.** Sometimes, the paper prescribes the use of more tricks than are strictly necessary, so be a bit wary of this, and try out simplifications where possible. For example, the original DDPG paper suggests a complex neural network architecture and initialization scheme, as well as batch normalization. These aren't strictly necessary, and some of the best-reported results for DDPG use simpler networks. As another example, the original A3C paper uses asynchronous updates from the various actor-learners, but it turns out that synchronous updates work about as well. 

**Don't overfit to existing implementations either.** Study `existing`_ `implementations`_ for inspiration, but be careful not to overfit to the engineering details of those implementations. RL libraries frequently make choices for abstraction that are good for code reuse between algorithms, but which are unnecessary if you're only writing a single algorithm or supporting a single use case.

**Iterate fast in simple environments.** To debug your implementations, try them with simple environments where learning should happen quickly, like CartPole-v0, InvertedPendulum-v0, FrozenLake-v0, and HalfCheetah-v2 (with a short time horizon---only 100 or 250 steps instead of the full 1000) from the `OpenAI Gym`_. Don’t try to run an algorithm in Atari or a complex Humanoid environment if you haven’t first verified that it works on the simplest possible toy task. Your ideal experiment turnaround-time at the debug stage is <5 minutes (on your local machine) or slightly longer but not much. These small-scale experiments don't require any special hardware, and can be run without too much trouble on CPUs.  

**If it doesn't work, assume there's a bug.** Spend a lot of effort searching for bugs before you resort to tweaking hyperparameters: usually it’s a bug. Bad hyperparameters can significantly degrade RL performance, but if you're using hyperparameters similar to the ones in papers and standard implementations, those will probably not be the issue. Also worth keeping in mind: sometimes things will work in one environment even when you have a breaking bug, so make sure to test in more than one environment once your results look promising.  

**Measure everything.** Do a lot of instrumenting to see what’s going on under-the-hood. The more stats about the learning process you read out at each iteration, the easier it is to debug---after all, you can’t tell it’s broken if you can’t see that it’s breaking. I personally like to look at the mean/std/min/max for cumulative rewards, episode lengths, and value function estimates, along with the losses for the objectives, and the details of any exploration parameters (like mean entropy for stochastic policy optimization, or current epsilon for epsilon-greedy as in DQN). Also, watch videos of your agent’s performance every now and then; this will give you some insights you wouldn’t get otherwise.  

**Scale experiments when things work.** After you have an implementation of an RL algorithm that seems to work correctly in the simplest environments, test it out on harder environments. Experiments at this stage will take longer---on the order of somewhere between a few hours and a couple of days, depending. Specialized hardware---like a beefy GPU or a 32-core machine---might be useful at this point, and you should consider looking into cloud computing resources like AWS or GCE.

**Keep these habits!** These habits are worth keeping beyond the stage where you’re just learning about deep RL---they will accelerate your research!

开展一个研究项目
=============================

Once you feel reasonably comfortable with the basics in deep RL, you should start pushing on the boundaries and doing research. To get there, you'll need an idea for a project.

**Start by exploring the literature to become aware of topics in the field.** There are a wide range of topics you might find interesting: sample efficiency, exploration, transfer learning, hierarchy, memory, model-based RL, meta learning, and multi-agent, to name a few. If you're looking for inspiration, or just want to get a rough sense of what's out there, check out Spinning Up's `key papers <../spinningup/keypapers.html>`_ list. Find a paper that you enjoy on one of these subjects---something that inspires you---and read it thoroughly. Use the related work section and citations to find closely-related papers and do a deep dive in the literature. You’ll start to figure out where the unsolved problems are and where you can make an impact. 

**Approaches to idea-generation:** There are a many different ways to start thinking about ideas for projects, and the frame you choose influences how the project might evolve and what risks it will face. Here are a few examples:

**Frame 1: Improving on an Existing Approach.** This is the incrementalist angle, where you try to get performance gains in an established problem setting by tweaking an existing algorithm. Reimplementing prior work is super helpful here, because it exposes you to the ways that existing algorithms are brittle and could be improved. A novice will find this the most accessible frame, but it can also be worthwhile for researchers at any level of experience. While some researchers find incrementalism less exciting, some of the most impressive achievements in machine learning have come from work of this nature.

Because projects like these are tied to existing methods, they are by nature narrowly scoped and can wrap up quickly (a few months), which may be desirable (especially when starting out as a researcher). But this also sets up the risks: it's possible that the tweaks you have in mind for an algorithm may fail to improve it, in which case, unless you come up with more tweaks, the project is just over and you have no clear signal on what to do next.

**Frame 2: Focusing on Unsolved Benchmarks.** Instead of thinking about how to improve an existing method, you aim to succeed on a task that no one has solved before. For example: achieving perfect generalization from training levels to test levels in the `Sonic domain`_ or `Gym Retro`_. When you hammer away at an unsolved task, you might try a wide variety of methods, including prior approaches and new ones that you invent for the project. It is possible for a novice to approch this kind of problem, but there will be a steeper learning curve. 

Projects in this frame have a broad scope and can go on for a while (several months to a year-plus). The main risk is that the benchmark is unsolvable without a substantial breakthrough, meaning that it would be easy to spend a lot of time without making any progress on it. But even if a project like this fails, it often leads the researcher to many new insights that become fertile soil for the next project.


**Frame 3: Create a New Problem Setting.** Instead of thinking about existing methods or current grand challenges, think of an entirely different conceptual problem that hasn't been studied yet. Then, figure out how to make progress on it. For projects along these lines, a standard benchmark probably doesn't exist yet, and you will have to design one. This can be a huge challenge, but it’s worth embracing---great benchmarks move the whole field forward.

Problems in this frame come up when they come up---it's hard to go looking for them. 

**Avoid reinventing the wheel.** When you come up with a good idea that you want to start testing, that’s great! But while you’re still in the early stages with it, do the most thorough check you can to make sure it hasn’t already been done. It can be pretty disheartening to get halfway through a project, and only then discover that there's already a paper about your idea. It's especially frustrating when the work is concurrent, which happens from time to time! But don’t let that deter you---and definitely don’t let it motivate you to plant flags with not-quite-finished research and over-claim the merits of the partial work. Do good research and finish out your projects with complete and thorough investigations, because that’s what counts, and by far what matters most in the long run.


做严谨的强化学习研究
=============================

Now you’ve come up with an idea, and you’re fairly certain it hasn’t been done. You use the skills you’ve developed to implement it and you start testing it out on standard domains. It looks like it works! But what does that mean, and how well does it have to work to be important? This is one of the hardest parts of research in deep RL. In order to validate that your proposal is a meaningful contribution, you have to rigorously prove that it actually gets a performance benefit over the strongest possible baseline algorithm---whatever currently achieves SOTA (state of the art) on your test domains. If you’ve invented a new test domain, so there’s no previous SOTA, you still need to try out whatever the most reliable algorithm in the literature is that could plausibly do well in the new test domain, and then you have to beat that. 

**Set up fair comparisons.** If you implement your baseline from scratch---as opposed to comparing against another paper’s numbers directly---it’s important to spend as much time tuning your baseline as you spend tuning your own algorithm. This will make sure that comparisons are fair. Also, do your best to hold “all else equal” even if there are substantial differences between your algorithm and the baseline. For example, if you’re investigating architecture variants, keep the number of model parameters approximately equal between your model and the baseline. Under no circumstances handicap the baseline! It turns out that the baselines in RL are pretty strong, and getting big, consistent wins over them can be tricky or require some good insight in algorithm design. 

**Remove stochasticity as a confounder.** Beware of random seeds making things look stronger or weaker than they really are, so run everything for many random seeds (at least 3, but if you want to be thorough, do 10 or more). This is really important and deserves a lot of emphasis: deep RL seems fairly brittle with respect to random seed in a lot of common use cases. There’s potentially enough variance that two different groups of random seeds can yield learning curves with differences so significant that they look like they don’t come from the same distribution at all (see `figure 10 here`_).

**Run high-integrity experiments.** Don’t just take the results from the best or most interesting runs to use in your paper. Instead, launch new, final experiments---for all of the methods that you intend to compare (if you are comparing against your own baseline implementations)---and precommit to report on whatever comes out of that. This is to enforce a weak form of `preregistration`_: you use the tuning stage to come up with your hypotheses, and you use the final runs to come up with your conclusions.

**Check each claim separately.** Another critical aspect of doing research is to run an ablation analysis. Any method you propose is likely to have several key design decisions---like architecture choices or regularization techniques, for instance---each of which could separately impact performance. The claim you'll make in your work is that those design decisions collectively help, but this is really a bundle of several claims in disguise: one for each such design element. By systematically evaluating what would happen if you were to swap them out with alternate design choices, or remove them entirely, you can figure out how to correctly attribute credit for the benefits your method confers. This lets you make each separate claim with a measure of confidence, and increases the overall strength of your work. 

别想太多
================

Deep RL is an exciting, fast-moving field, and we need as many people as possible to go through the open problems and make progress on them. Hopefully, you feel a bit more prepared to be a part of it after reading this! And whenever you’re ready, `let us know`_.

.. _`let us know`: https://jobs.lever.co/openai


后记：其他资源
===================

Consider reading through these other informative articles about growing as a researcher or engineer in this field:

`Advice for Short-term Machine Learning Research Projects <https://rockt.github.io/2018/08/29/msc-advice>`_, by Tim Rocktäschel, Jakob Foerster and Greg Farquhar.

`ML Engineering for AI Safety & Robustness: a Google Brain Engineer’s Guide to Entering the Field <https://80000hours.org/articles/ml-engineering-career-transition-guide/>`_, by Catherine Olsson and 80,000 Hours. 

参考
==========

.. _`it’s hard and it doesn’t always work`: https://www.alexirpan.com/2018/02/14/rl-hard.html
.. [1] `不总是奏效 <https://www.alexirpan.com/2018/02/14/rl-hard.html>`_, Alex Irpan, 2018

.. _`可重复性`: https://arxiv.org/abs/1708.04133
.. _`figure 10 here`: https://arxiv.org/pdf/1708.04133.pdf
.. [2] `Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control <https://arxiv.org/abs/1708.04133>`_, Islam et al, 2017

.. _`is a challenge`: https://arxiv.org/abs/1709.06560
.. [3] `Deep Reinforcement Learning that Matters <https://arxiv.org/abs/1709.06560>`_, Henderson et al, 2017

.. _`the learning curve is incredibly steep`: http://amid.fish/reproducing-deep-rl
.. [4] `Lessons Learned Reproducing a Deep Reinforcement Learning Paper <http://amid.fish/reproducing-deep-rl>`_, Matthew Rahtz, 2018

.. _`great`: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
.. [5] `UCL Course on RL <http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html>`_

.. _`resources`: http://rll.berkeley.edu/deeprlcourse/
.. [6] `Berkeley Deep RL Course <http://rll.berkeley.edu/deeprlcourse/>`_

.. _`out`: https://sites.google.com/view/deep-rl-bootcamp/lectures
.. [7] `Deep RL Bootcamp <https://sites.google.com/view/deep-rl-bootcamp/lectures>`_

.. _`there`: http://joschu.net/docs/nuts-and-bolts.pdf
.. [8] `Nuts and Bolts of Deep RL <http://joschu.net/docs/nuts-and-bolts.pdf>`_, John Schulman

.. _`MLP`: http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
.. [9] `Stanford Deep Learning Tutorial: Multi-Layer Neural Network <http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/>`_

.. _`Vanilla RNN`: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
.. [10] `The Unreasonable Effectiveness of Recurrent Neural Networks <http://karpathy.github.io/2015/05/21/rnn-effectiveness/>`_, Andrej Karpathy, 2015

.. _`LSTM`: https://arxiv.org/abs/1503.04069
.. [11] `LSTM: A Search Space Odyssey <https://arxiv.org/abs/1503.04069>`_, Greff et al, 2015

.. _`also see this blog`: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
.. [12] `Understanding LSTM Networks <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_, Chris Olah, 2015

.. _`GRU`: https://arxiv.org/abs/1412.3555v1
.. [13] `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling <https://arxiv.org/abs/1412.3555v1>`_, Chung et al, 2014 (GRU paper)

.. _`conv`: http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
.. [14] `Conv Nets: A Modular Perspective <http://colah.github.io/posts/2014-07-Conv-Nets-Modular/>`_, Chris Olah, 2014

.. _`layers`: https://cs231n.github.io/convolutional-networks/
.. [15] `Stanford CS231n, Convolutional Neural Networks for Visual Recognition <https://cs231n.github.io/convolutional-networks/>`_

.. _`resnets`: https://arxiv.org/abs/1512.03385
.. [16] `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_, He et al, 2015 (ResNets)

.. _`attention`: https://arxiv.org/abs/1409.0473
.. [17] `Neural Machine Translation by Jointly Learning to Align and Translate <https://arxiv.org/abs/1409.0473>`_, Bahdanau et al, 2014 (Attention mechanisms)

.. _`mechanisms`: https://arxiv.org/abs/1706.03762
.. [18] `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_, Vaswani et al, 2017

.. _`weight decay`: https://papers.nips.cc/paper/563-a-simple-weight-decay-can-improve-generalization.pdf
.. [19] `A Simple Weight Decay Can Improve Generalization <https://papers.nips.cc/paper/563-a-simple-weight-decay-can-improve-generalization.pdf>`_, Krogh and Hertz, 1992


.. _`dropout`: http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf
.. [20] `Dropout:  A Simple Way to Prevent Neural Networks from Overfitting <http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf>`_, Srivastava et al, 2014

.. _`batch norm`: https://arxiv.org/abs/1502.03167
.. [21] `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_, Ioffe and Szegedy, 2015

.. _`layer norm`: https://arxiv.org/abs/1607.06450
.. [22] `Layer Normalization <https://arxiv.org/abs/1607.06450>`_, Ba et al, 2016

.. _`weight norm`: https://arxiv.org/abs/1602.07868
.. [23] `Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks <https://arxiv.org/abs/1602.07868>`_, Salimans and Kingma, 2016

.. _`SGD, momentum SGD`: http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/
.. [24] `Stanford Deep Learning Tutorial: Stochastic Gradient Descent <http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/>`_

.. _`Adam`: https://arxiv.org/abs/1412.6980
.. [25] `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_, Kingma and Ba, 2014

.. _`others`: https://arxiv.org/abs/1609.04747
.. [26] `An overview of gradient descent optimization algorithms <https://arxiv.org/abs/1609.04747>`_, Sebastian Ruder, 2016

.. _`reparameterization trick`: https://arxiv.org/abs/1312.6114
.. [27] `Auto-Encoding Variational Bayes <https://arxiv.org/abs/1312.6114>`_, Kingma and Welling, 2013 (Reparameterization trick)

.. _`Tensorflow`: https://www.tensorflow.org/
.. [28] `Tensorflow`_

.. _`PyTorch`: http://pytorch.org/
.. [29] `PyTorch`_

.. _`an introduction`: ../spinningup/rl_intro.html
.. [30] `Spinning Up in Deep RL: Introduction to RL, Part 1 <../spinningup/rl_intro.html>`_

.. _`RL-Intro`: https://github.com/jachiam/rl-intro/blob/master/Presentation/rl_intro.pdf
.. [31] `RL-Intro`_ Slides from OpenAI Hackathon, Josh Achiam, 2018

.. _`overview by Lilian Weng`: https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html
.. [32] `A (Long) Peek into Reinforcement Learning <https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html>`_, Lilian Weng, 2018

.. _`monotonic improvement theory`: http://joschu.net/docs/thesis.pdf
.. [33] `Optimizing Expectations <http://joschu.net/docs/thesis.pdf>`_, John Schulman, 2016 (Monotonic improvement theory)

.. _`classical RL algorithms`: https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf
.. [34] `Algorithms for Reinforcement Learning <https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf>`_, Csaba Szepesvari, 2009 (Classic RL Algorithms)

.. _`REINFORCE`: https://arxiv.org/abs/1604.06778
.. [35] `Benchmarking Deep Reinforcement Learning for Continuous Control <https://arxiv.org/abs/1604.06778>`_, Duan et al, 2016

.. _`DQN`: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
.. [36] `Playing Atari with Deep Reinforcement Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_, Mnih et al, 2013 (DQN)

.. _`A2C`: https://blog.openai.com/baselines-acktr-a2c/
.. [37] `OpenAI Baselines: ACKTR & A2C <https://blog.openai.com/baselines-acktr-a2c/>`_

.. _`A3C`: https://arxiv.org/abs/1602.01783
.. [38] `Asynchronous Methods for Deep Reinforcement Learning <https://arxiv.org/abs/1602.01783>`_, Mnih et al, 2016 (A3C)

.. _`PPO`: https://arxiv.org/abs/1707.06347
.. [39] `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_, Schulman et al, 2017 (PPO)

.. _`DDPG`: https://arxiv.org/abs/1509.02971
.. [40] `Continuous Control with Deep Reinforcement Learning <https://arxiv.org/abs/1509.02971>`_, Lillicrap et al, 2015 (DDPG)

.. _`a no-frills version of VPG`: https://github.com/jachiam/rl-intro/blob/master/pg_cartpole.py
.. [41] `RL-Intro Policy Gradient Sample Code <https://github.com/jachiam/rl-intro/blob/master/pg_cartpole.py>`_, Josh Achiam, 2018

.. _`existing`: https://github.com/openai/baselines
.. [42] `OpenAI Baselines <https://github.com/openai/baselines>`_

.. _`implementations`: https://github.com/rll/rllab
.. [43] `rllab <https://github.com/rll/rllab>`_

.. _`OpenAI Gym`: https://gym.openai.com/
.. [44] `OpenAI Gym <https://gym.openai.com/>`_

.. _`Sonic domain`: https://contest.openai.com/2018-1/
.. [45] `OpenAI Retro Contest <https://contest.openai.com/2018-1/>`_

.. _`Gym Retro`: https://blog.openai.com/gym-retro/
.. [46] `OpenAI Gym Retro <https://blog.openai.com/gym-retro/>`_

.. _`preregistration`: https://cos.io/prereg/
.. [47] `Center for Open Science <https://cos.io/prereg/>`_, explaining what preregistration means in the context of scientific experiments.