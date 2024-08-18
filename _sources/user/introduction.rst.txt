============
项目介绍
============

.. contents:: 目录

这个项目是什么
============

欢迎来到深度强化学习（deep Reinforement Learning）的Spinning Up项目！这是一份由OpenAI提供的教育资源，旨在让深度强化学习的学习变得更加简单。

`强化学习`_ ，是一种通过教会智能体（agents）反复试错从而完成任务的机器学习方法。深度强化学习指的是强化学习和 `深度学习`_ 的结合。

这个模块包括一系列有用的资源，包括：

- 关于强化学习术语、算法和基础理论的简单 `介绍`_
- 关于如何成为强化学习的研究人员的 `文章`_
- 一个按照主题整理的重要论文 `清单`_
- 以及一些用来练手的 `项目`_


.. _`强化学习`: https://en.wikipedia.org/wiki/Reinforcement_learning
.. _`深度学习`: http://ufldl.stanford.edu/tutorial/

为什么创建这个项目
=================

我们最常听到的问题是：

    | 如果我想为AI安全做贡献，我应该如何开始？


在OpenAI，我们相信深度学习尤其是深度强化学习，会在未来的人工智能科技中扮演重要角色。为了确保人工智能的安全性，我们必须提出与此相契合的安全策略和算法。因此我们鼓励每一个提出这个问题的人研究这些领域。

深度学习现在有很多帮助人们快速入门的资源，相比之下，深度强化学习显得门槛高很多。首先，学生要有数学、编程和深度学习的背景。除此之外，他们需要对于这一领域有高水准的理解：都有哪些研究课题？这些课题为什么重要？哪些东西已经做出来了？他们也需要认真的指导，从而了解如何将算法理论和代码结合起来。


这个领域还很新，所以一时很难有高层次的了解。现在深度强化学习领域还没有合适的教材，所以很多知识都被藏在了论文和讲座中，难以理解吸收。深度强化学习算法实现的学习也很痛苦，因为：

- 很多算法论文或有意或无意的省略了核心的细节部分
- 一些知名度很高的算法实现代码很难读懂，难以把代码和算法联系一块

尽管很多很棒的 repos ，例如 rllab_, Baselines_, 和 rllib_ ，让那些已经在这个领域的研究者更加容易做出成果。但这些项目会考虑很多因素综合权衡之后，把算法整合到框架里面，代码也就变得难以看懂。所以说，不管是对于学者还是从业者、业余爱好者来说，深度强化学习领域的门槛都很高。

我们的项目就是为了填上中间缺的这一部分，服务于那些，希望了解深度强化学习或者希望做出自己的贡献，但是对于要学习什么内容以及如果把算法变成代码不清楚的同学。我们努力把这个项目变成一个助推器。

也就是说，从业人员不是这个项目唯一的受益者.人工智能安全问题的解决，不仅需要人们有大量实践经验和广阔的视野，还需要了解很多与工程、计算机无关的专业知识。每一个参与到这个计算的人都应该做出精明的决定，Spinning Up 项目的很多地方都提到了这一点。

这个项目如何服务我们的使命
===========================

OpenAI 的 `使命`_ 是确保通用人工智能(Artificial general intelligence, AGI)的安全发展以及让人工智能带来的收益分布更加均匀。Spinning Up 这样的教育工具能够在这两个方面都作出贡献。

只要我们能帮助更多人了解人工智能究竟是什么以及它是怎么工作的，我们就能更接近广泛的利益分配。这会促使人们批判地思考很多问题，因为人工智能在我们的生活中变得越来越重要。

同时，我们也需要人 `加入我们 <https://jobs.lever.co/openai>`_ 共同确保通用人工智能的安全。由于这一领域还很新，所以拥有这项技能的人才目前供不应求。如果您能通过这个项目成为专家，那你一定能在我们的人工智能安全上发挥作用。

代码设计的原则
======================

Spinning Up 项目的算法实现的时候有下面两个要求：

    - 尽量简单，同时还要足够好
    - 不同算法实现之间高度一致，从而揭示他们之间的相似性

这些算法基本上都是相互独立的，没有相互依赖的代码（除了日志打印、保存、载入和 `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ 等工具模块），所以你可以独立的学习每一个算法而不用去管那些繁琐的依赖关系。这些实现尽量做到和伪代码一致，最小化理论和代码之间的差距。

他们的结构都很相似，所以如果你理解了一个算法，再看剩下的就很简单。

我们尽量减少算法实现时候的技巧(trick)和相似算法之间的区别。这里可以展示一些移除的技巧，我们把原始 Soft-Actor Critic 算法中的 正则_ 和 `观察正则化(observation normalization)`_ 都移除了。我们在 DDPG, TD3, 和 SAC 的实现中，都遵循了 `原始TD3代码`_ 的约定，所有的梯度更新都是在每一个回合的最后执行的（而不是整个回合都在执行）。

所有的算法都做到“足够好”是指性能大致达到最优，但不一定达到了最优效果。所以进行科研基准(benchmark)的对比时要小心。每种实现的性能表现可以在我们的 基准_ 找到。

支持计划
============

我们计划支持 Spinning Up 项目来确保他能够作为学习深度强化学习的实用资料。这个项目的长期支持（数年内）还没有确定，但是短期内我们可以承诺:

- 发布后的前三周会大力支持（2018年11月8日至2018年11月29日）

    + 我们会通过修复 bug ，回答问题和改善文档以修复歧义的方式快速迭代
    + 我们会努力提升用户体验，方便用户使用该项目自助学习。


- 发布 6 个月之后，我们会根据社区反馈对整个项目做评估，然后宣布未来的计划，包括长期的规划路线。

此外，正如博客文章讨论的，我们也会在即将到来的 Scholars_ 和 Fellows_ 课程中使用 Spinning Up。任何更改和更新都会实时同步公开。


.. _`介绍`: ../spinningup/rl_intro.html
.. _`文章`: ../spinningup/spinningup.html
.. _`Spinning Up essay`: ../spinningup/spinningup.html
.. _`清单`: ../spinningup/keypapers.html
.. _`code repo`: https://github.com/openai/spinningup
.. _`项目`: ../spinningup/exercises.html
.. _`rllab`: https://github.com/rll/rllab
.. _`Baselines`: https://github.com/openai/baselines
.. _`rllib`: https://github.com/ray-project/ray/tree/master/python/ray/rllib
.. _`使命`: https://blog.openai.com/openai-charter/
.. _`正则`: https://github.com/haarnoja/sac/blob/108a4229be6f040360fcca983113df9c4ac23a6a/sac/distributions/normal.py#L69
.. _`观察正则化(observation normalization)`: https://github.com/openai/baselines/blob/28aca637d0f13f4415cc5ebb778144154cff3110/baselines/run.py#L131
.. _`original TD3 code`: https://github.com/sfujim/TD3/blob/25dfc0a6562c54ae5575fad5b8f08bc9d5c4e26c/main.py#L89
.. _`基准`: ../spinningup/bench.html
.. _Scholars : https://jobs.lever.co/openai/cf6de4ed-4afd-4ace-9273-8842c003c842
.. _Fellows : https://jobs.lever.co/openai/c9ba3f64-2419-4ff9-b81d-0526ae059f57


