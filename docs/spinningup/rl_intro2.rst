==================================
第二部分：强化学习算法概述
==================================

.. contents:: 目录
    :depth: 2

我们已经介绍了强化学习的术语和符号表示，现在可以讨论一些更丰富的内容：现代强化学习的整体发展和算法设计时候要考虑的各种因素之间的权衡。

强化学习算法的分类（A Taxonomy of RL Algorithms）
==============================================

.. figure:: ../images/rl_algorithms_9_15.svg
    :align: center

    一个不是很详细但是十分有用的现代强化学习算法分类。  `参见`_

要先声明的是：很难准确全面的把所有现代强化学习算法都列举出来，因为这些内容本身不适合用树形结构展示。同时，把这么多内容放在一篇文章里，还要便于理解，必须省略掉一些更加高深的内容，例如探索（exploration），迁移学习（transfer learning），元学习（meta learning）等。

这篇文章的目标是：

* 只强调深度强化学习中关于学习什么和如何学习的最基础的设计选择
* 揭示这些选择中的利弊权衡
* 把其中部分优秀的现代算法介绍给大家

无模型学习（Model-Free） vs 有模型学习（Model-Based）的强化学习
-----------------------------------------------------------

不同强化学习算法最重要的区分点之一就是 **智能体是否能完整了解或学习到所在环境的模型**。 环境的模型是指一个预测状态转换和奖励的函数。

有模型最大的优势在于智能体能够 **提前考虑来进行规划**，走到每一步的时候，都提前尝试未来可能的选择，然后明确地从这些候选项中进行选择。智能体可以把预先规划的结果提取为学习策略。这其中最著名的例子就是 `AlphaZero`_。这个方法起作用的时候，可以大幅度提升采样效率（sample efficiency） —— 相对于那些没有模型的方法。

有模型最大的缺点就是智能体往往不能获得环境的真实模型。如果智能体想在一个场景下使用模型，那它必须完全从经验中学习，这会带来很多挑战。最大的挑战就是，智能体探索出来的模型和真实模型之间存在误差，而这种误差会导致智能体在学习到的模型中表现很好，但在真实的环境中表现得不好（甚至很差）。基于模型的学习从根本上讲是非常困难的，即使你愿意花费大量的时间和计算力，最终的结果也可能达不到预期的效果。

使用模型的算法叫做有模型学习，不基于模型的叫做无模型学习。虽然无模型学习放弃了有模型学习在样本效率方面的潜在收益，但是他们往往更加易于实现和调整。截止到目前（2018年9月），相对于有模型学习，无模型学习方法更受欢迎，得到更加广泛的开发和测试。

要学习什么
-------------

强化学习算法另一个重要的分支是 **要学习什么**。常提到的主题包括：

* 策略，不管是随机的还是确定性的
* 动作-值函数（Q 函数）
* 值函数
* 环境模型

无模型学习中要学习什么
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

有两种主要的方法使用无模型学习的强化学习算法来表示和训练智能体：

**策略优化（Policy Optimization）** ：这类方法将策略显示表示为： :math:`\pi_{\theta}(a|s)` 。 它们直接对性能目标 :math:`J(\pi_{\theta})` 进行梯度下降进行优化，或者间接地，对性能目标的局部近似函数进行优化。优化基本都是基于 **on-policy** 的，也就是说每一步更新只会用最新的策略执行时采集到的数据。策略优化通常还包括学习出 :math:`V_{\phi}(s)` ，作为 :math:`V^{\pi}(s)` 的近似，该函数用于确定如何更新策略。

基于策略优化的方法举例：

* `A2C / A3C`_, 通过梯度下降直接最大化性能，
* `PPO`_ , 间接通过最大化性能，通过最大化 **代理目标** （surrogate objective） 函数，这个函数是目标函数 :math:`J(\pi_{\theta})` 的近似估计。

**Q-Learning** ：这个系列的算法学习最优动作-值函数 :math:`Q^*(s,a)` 的近似函数： :math:`Q_{\theta}(s,a)` 。它们通常使用基于 `贝尔曼方程`_ 的目标函数。优化过程属于 **off-policy** ，这意味着每次更新可以使用任意时间点的训练数据，不管获取数据时智能体选择如何探索环境。对应的策略是通过  :math:`Q^*` and :math:`\pi^*` 之间的联系得到的。智能体的行动由下面的式子给出：

.. math::
    
    a(s) = \arg \max_a Q_{\theta}(s,a).

基于 Q-Learning 的方法

* `DQN`_, 一个让深度强化学习得到发展的经典方法，
* 以及 `C51`_, 学习关于回报的分布函数，其期望是 :math:`Q^*` 

**策略优化和 Q-Learning 的权衡** ：策略优化的主要优势在于这类方法是原则性的，某种意义上讲，你是直接在优化你想要的东西，这使得他们更加稳定和可靠。与此相反，Q-learning 方法通过训练 :math:`Q_{\theta}` 以满足自洽方程，间接地优化智能体的表现。这种方法有很多失败的情况，所以相对来说稳定性较差。[1]_ 但是，Q-learning 有效的时候能获得更好的采样效率，因为它们能够比策略优化更加有效地重新使用训练数据。

**策略优化和 Q-learning 的融合方法** ：意外的是，策略优化和 Q-learning 并不是不能兼容的（在某些场景下，它们两者是 `等价的`_ ），并且存在很多介于两种极端之间的算法。这个范围的算法能够很好的平衡好两者之间的优点和缺点，比如说：

* `DDPG`_ 是一种同时学习确定性策略和 Q 函数的算法
* `SAC`_ 是一种变体，它使用随机策略、交叉熵正则化和一些其它技巧来稳定学习，同时在 benchmarks 上获得比 DDPG 更高的分数。

.. [1] 关于更多 Q-learning 可能会表现不好的情况，参见： 1) 经典论文 `Tsitsiklis and van Roy`_, 2) 最近的文章 `review by Szepesvari`_ (在 4.3.2章节)  3) `Sutton and Barto`_ 的第11章节，尤其是 11.3 (on "the deadly triad" of function approximation, bootstrapping, and off-policy data, together causing instability in value-learning algorithms).

.. _`贝尔曼方程`: ../spinningup/rl_intro.html#bellman-equations
.. _`Tsitsiklis and van Roy`: http://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf
.. _`review by Szepesvari`: https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf
.. _`Sutton and Barto`: https://drive.google.com/file/d/1xeUDVGWGUUv1-ccUMAZHJLej2C7aAFWY/view
.. _`等价的`: https://arxiv.org/abs/1704.06440

有模型强化学习要学习什么
-------------------------------

不同于无模型学习，有模型学习方法不是很好分类：很多方法之间都会有交叉。我们会给出一些例子，当然肯定不够详尽，覆盖不到全部。在这些例子里面， **模型** 要么已知，要么是可学习的。

**背景：纯规划（Pure Planning）** ：这种最基础的方法，从来不显示的使用策略，而是纯使用规划技术来选择行动，例如 `模型预测控制`_ (model-predictive control, MPC)。在模型预测控制中，智能体每次观察环境的时候，都会计算得到一个对于当前模型最优的规划，这里的规划指的是未来一个固定时间段内，智能体会采取的所有行动（通过学习值函数，规划算法可能会考虑到超出范围的未来奖励）。智能体先执行规划的第一个行动，然后立即舍弃规划的剩余部分。每次准备和环境进行互动时，它会计算出一个新的规划，从而避免执行小于规划范围的规划给出的行动。

* `MBMF`_ 在一些深度强化学习的标准基准任务上，基于学习到的环境模型进行模型预测控制。

**Expert Iteration** ：纯规划的后来之作，使用、学习策略的显示表示形式： :math:`\pi_{\theta}(a|s)` 。智能体在模型中应用了一种规划算法，类似蒙特卡洛树搜索(Monte Carlo Tree Search)，通过对当前策略进行采样生成规划的候选行为。这种算法得到的行动比策略本身生成的要好，所以相对于策略来说，它是“专家”。随后更新策略，以产生更类似于规划算法输出的行动。

* `ExIt`_ 算法用这种算法训练深层神经网络来玩 Hex，
* `AlphaZero`_ 这种方法的另一个例子。

**无模型方法的数据增强** ：使用无模型算法来训练策略或者 Q 函数，要么 1）更新智能体的时候，用构造出的假数据来增加真实经验 2）更新的时候 **仅** 使用构造的假数据。

* `MBVE`_  用假数据增加真实经验，
* `World Models`_ 全部用假数据来训练智能体，所以被称为：“在梦里训练”。

**Embedding Planning Loops into Policies.**  ：另一种方法直接把规划程序作为策略的子程序，这样在基于任何无模型算法训练策略输出的时候，整个规划就变成了策略的附属信息。这个框架最核心的概念就是，策略可以学习到如何以及何时使用规划。这使得模型偏差不再是问题，因为如果模型在某些状态下不利于规划，那么策略可以简单地学会忽略它。

* 更多例子，参见 `I2A`_ 

.. _`模型预测控制`: https://en.wikipedia.org/wiki/Model_predictive_control
.. _`ExIt`: https://arxiv.org/abs/1705.08439
.. _`World Models`: https://worldmodels.github.io/



分类中提到的算法链接
===============================

.. _`参见`: 

.. [#] `A2C / A3C <https://arxiv.org/abs/1602.01783>`_ (Asynchronous Advantage Actor-Critic): Mnih et al, 2016
.. [#] `PPO <https://arxiv.org/abs/1707.06347>`_ (Proximal Policy Optimization): Schulman et al, 2017 
.. [#] `TRPO <https://arxiv.org/abs/1502.05477>`_ (Trust Region Policy Optimization): Schulman et al, 2015
.. [#] `DDPG <https://arxiv.org/abs/1509.02971>`_ (Deep Deterministic Policy Gradient): Lillicrap et al, 2015
.. [#] `TD3 <https://arxiv.org/abs/1802.09477>`_ (Twin Delayed DDPG): Fujimoto et al, 2018
.. [#] `SAC <https://arxiv.org/abs/1801.01290>`_ (Soft Actor-Critic): Haarnoja et al, 2018
.. [#] `DQN <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_ (Deep Q-Networks): Mnih et al, 2013
.. [#] `C51 <https://arxiv.org/abs/1707.06887>`_ (Categorical 51-Atom DQN): Bellemare et al, 2017
.. [#] `QR-DQN <https://arxiv.org/abs/1710.10044>`_ (Quantile Regression DQN): Dabney et al, 2017
.. [#] `HER <https://arxiv.org/abs/1707.01495>`_ (Hindsight Experience Replay): Andrychowicz et al, 2017
.. [#] `World Models`_: Ha and Schmidhuber, 2018
.. [#] `I2A <https://arxiv.org/abs/1707.06203>`_ (Imagination-Augmented Agents): Weber et al, 2017 
.. [#] `MBMF <https://sites.google.com/view/mbmf>`_ (Model-Based RL with Model-Free Fine-Tuning): Nagabandi et al, 2017 
.. [#] `MBVE <https://arxiv.org/abs/1803.00101>`_ (Model-Based Value Expansion): Feinberg et al, 2018
.. [#] `AlphaZero <https://arxiv.org/abs/1712.01815>`_: Silver et al, 2017 


