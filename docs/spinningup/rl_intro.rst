==========================
第一部分：强化学习中的核心概念
==========================


.. contents:: 目录
    :depth: 2

欢迎来到强化学习的介绍部分！我们希望你能了解以下内容：

* 这部分讨论用到的数学符号表示
* 关于强化学习算法做什么的一个深层解释（我们会尽量避免提及 *他们是如何做到的* 这个话题）
* 以及算法背后的一些核心数学知识

总的来说，强化学习是关于智能体（agents）以及它们如何通过试错来学习的研究。达成了这样一个共识：通过奖励或惩罚智能体（agents）的行为从而使它未来更容易重复或者放弃某一行为。

强化学习能做什么
===============

基于强化学习的方法已经在很多地方取得了成功。例如，它被用来教计算机在仿真环境下控制机器人：

.. raw:: html

    <video autoplay="" src="https://storage.googleapis.com/joschu-public/knocked-over-stand-up.mp4" loop="" controls="" style="display: block; margin-left: auto; margin-right: auto; margin-bottom:1.5em; width: 100%; max-width: 720px; max-height: 80vh;">
    </video>

以及在现实世界中控制机器：

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/jwSbzNHGflM?ecver=1" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>
    <br />

强化学习因为被用在复杂策略游戏创造出突破性的 AI 中而名声大噪，最著名的要数 `围棋`_ 、 `Dota`_ 、教计算机 `玩Atari游戏`_ 以及训练模拟机器人 `听从人类的指令`_ 。

.. _`围棋`: https://deepmind.com/research/alphago/
.. _`Dota`: https://blog.openai.com/openai-five/
.. _`玩Atari游戏`: https://deepmind.com/research/dqn/
.. _`听从人类的指令`: https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/


核心概念和术语
============================

.. figure:: ../images/rl_diagram_transparent_bg.png
    :align: center
    
    智能体和环境的循环交互

强化学习的主要特征是 **智能体** （agents）和 **环境**（environment），环境是智能体生存和互动的世界。在每一步的交互中，智能体都能得到一个这个世界的（有可能只是一部分）状态（state）的观察（observation），然后决定要采取的动作。环境会因为智能体对它的动作（actions）而改变，也可能自己改变。

智能体也会从环境中感知到 **奖励**（reward） 信号，一个表明当前世界状态好坏的数字。智能体的目标是最大化它获得的累计奖励，也就是 **回报**（return）。强化学习方法就是智能体通过学习行为来完成目标的方式。

为了更具体地讨论强化学习的作用，我们需要引入一些的术语：

* 状态和观察(states and observations)
* 动作空间(action spaces)
* 策略(policies)
* 轨迹(trajectories)
* 不同的回报公式(different formulations of return)
* 强化学习优化问题(the RL optimization problem)
* 和值函数(value functions)

状态和观察（States and Observations）
------------------------------------

一个 **状态** :math:`s` 是一个关于这个世界状态的完整描述。这个世界所有的信息都包含在状态中。**观察** :math:`o` 是对于一个状态的部分描述，可能会漏掉一些信息。

在深度强化学习中，我们一般用 `实数向量、矩阵或者更高阶的张量（tensor）`_ 表示状态和观察。比如说，图像的 **观察** 可以用RGB矩阵的方式表示其像素值；机器人的 **状态** 可以通过关节角度和速度来表示。

如果智能体观察到环境的全部状态，我们通常说环境是被 **全面观察** （fully observed）的。如果智能体只能观察到一部分状态，我们称之为 **部分观察** （partially observed）。

.. admonition:: 你应该知道

    强化学习有时候用表示状态的符号 :math:`s` 放在一些适合使用符号 :math:`o` 的地方来表示观察.  尤其是，当智能体在决定采取什么动作的时候，符号上的表示按理说动作是基于当前状态的决定的，但实际上，因为智能体并不能知道状态所以动作是基于观察的。

    在我们的教程中，我们会按照标准的方式使用这些符号，不过你一般能从上下文中看出来具体表示什么。如果你觉得有些内容不够清楚，请提出issue！我们的目的是教会大家，不是让大家混淆。

.. _`实数向量、矩阵或者更高阶的张量（tensor）`: https://en.wikipedia.org/wiki/Real_coordinate_space

动作空间（Action Spaces）
--------------------------

不同的环境允许不同的动作。所有有效动作的集合称之为 **动作空间**。有些环境，比如说 Atari 游戏和围棋，属于 **离散动作空间**，这种情况下智能体只能采取有限的动作。其他的一些环境，比如智能体在物理世界中控制机器人，属于 **连续动作空间**。在连续动作空间中，动作是实数向量。

这种区别对于深度强化学习来说影响很大。有些算法只能直接用在某些某一种情况，如果需要想使用于另外的情况，可能就需要改进很多。

策略（Policies）
---------------

**策略** 是智能体用于决定下一步执行什么行动的规则。可以是确定性的，一般表示为：:math:`\mu`:

.. math::

    a_t = \mu(s_t),

也可以是随机的，一般表示为 :math:`\pi`:

.. math::

    a_t \sim \pi(\cdot | s_t).

因为策略本质上就是智能体的大脑，所以很多时候“策略”和“智能体”这两个名词经常混用，例如我们会说：“策略的目的是最大化奖励”。

在深度强化学习中，我们处理的是参数化的策略，这些策略的输出，依赖于一系列计算函数，而这些函数又依赖于参数（例如神经网络的权重和误差），所以我们可以通过一些优化算法改变智能体的的行为。

我们经常把这些策略的参数计为 :math:`\theta` 或 :math:`\phi` ，然后把它写在策略的下标上来强调两者的联系。

.. math::

    a_t &= \mu_{\theta}(s_t) \\
    a_t &\sim \pi_{\theta}(\cdot | s_t).

确定性策略（Deterministic Policies）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**例子：确定性策略：** 这是一个基于 PyTorch 使用 `torch.nn` 库在连续动作空间上构建一个确定性策略的简单例子：

.. code-block:: python

    pi_net = nn.Sequential(
              nn.Linear(obs_dim, 64),
              nn.Tanh(),
              nn.Linear(64, 64),
              nn.Tanh(),
              nn.Linear(64, act_dim)
            )

这里构建了一个多层感知器的网络，包含两个有大小为64的隐含层和`tanh`激活函数，如果`obs`是一个包含一批观测值的Numpy数组，`pi_net`能够使用来获得一批动作：

.. code-block:: python

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    actions = pi_net(obs_tensor)

.. admonition:: 你应该知道

    如果你对神经网络的内容不熟悉，也不要担心，本教程将侧重于强化学习，而不是神经网络方面的内容。因此，您可以跳过这个示例，稍后再回到它。但我们觉得如果你已经知道了，可能会有帮助。

随机性策略（Stochastic Policies）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

深度强化学习中最常见的两类随机策略是 **分类策略** (Categorical Policies) 和 **对角高斯策略** (Diagonal Gaussian Policies)。

`分类`_ 策略适用于离散行动空间，而 `高斯`_ 策略一般用在连续行动空间。

使用和训练随机策略的时候有两个重要的计算步骤是：

* 从策略中采样行动
* 计算给定行为的对数似然函数(log likelihoods) :math:`\log \pi_{\theta}(a|s)`.


在接下来的内容中，我们将描述如何使用分类策略和对角高斯策略实现这两个计算步骤。


.. admonition:: 分类策略

    分类策略就像是一个离散空间的分类器(classifier)。像建立一个分类器的神经网络一样建立一个分类策略的神经网络：输入是观察，接着是一些卷积、全连接层之类的，至于具体是哪些取决于输入的类型，最后一个线性层给出每个行动的 log 数值(logits)，后面跟一个 `softmax`_ 层把 log 数值转换为概率。   

    **采样** 已知每个行动的概率，PyTorch和Tensorflow之类的框架有内置函数可以进行采样。具体可查阅 `Categorical distributions`_, `torch.multinomial`_, `tf.distributions.Categorical`_ , 或 `tf.multinomial`_ 。

    **对数似然** ：最后一层的概率定义为 :math:`P_{\theta}(s)`。它是一个和动作数量相同的向量，我们可以把动作当做索引。所以动作 :math:`a` 对数似然值可以通过这样得到：


    .. math::

        \log \pi_{\theta}(a|s) = \log \left[P_{\theta}(s)\right]_a.


.. admonition:: 对角高斯策略

    多元高斯分布（或者多元正态分布），可以用一个向量 :math:`\mu` 和协方差 :math:`\Sigma` 来描述。对角高斯分布就是协方差矩阵只有对角线上有值的特殊情况，所以我们可以用一个向量来表示它。

    对角高斯策略总会有一个神经网络，表示观察到行动的映射。其中有两种协方差矩阵的经典表示方式：

    **第一种** ： 有一个单独的关于对数标准差的向量： :math:`\log \sigma`，它不是关于状态的函数，:math:`\log \sigma` 而是单独的参数（我们这个项目里，VPG, TRPO 和 PPO 都是用这种方式实现的）。

    **第二种** ：有一个神经网络，从状态映射到对数标准差 :math:`\log \sigma_{\theta}(s)`。这种方式可能会和均值网络共享某些层的参数。

    要注意这两种情况下我们都没有直接计算标准差而是计算了对数标准差。这是因为对数标准差的定义域是 :math:`(-\infty, \infty)` ，而标准差必须要求参数非负。约束条件越少，训练就越简单。而标准差可以通过对数标准差取幂得到，所以这种表示方法也不会丢失信息。

    **采样** ：给定平均动作  :math:`\mu_{\theta}(s)` 和 标准差 :math:`\sigma_{\theta}(s)`，以及一个服从球形高斯分布的噪声向量 :math:`z`，动作的样本可以这样计算：

    .. math::

        a = \mu_{\theta}(s) + \sigma_{\theta}(s) \odot z,

    这里 :math:`\odot` 表示两个向量的点积。标准框架都有内置函数生成噪音向量，例如  `torch.normal`_ 和 `tf.random_normal`_ 。你也可以直接内置分布例如 `torch.distributions.Normal`_ 或者 `tf.distributions.Normal`_ 采样(后者的优势是哪些分布函数可以直接为你计算对数似然)。

    **对数似然** 一个 k 维动作 :math:`a` 基于均值为 :math:`\mu = \mu_{\theta}(s)`，标准差为 :math:`\sigma = \sigma_{\theta}(s)` 的对角高斯的对数似然如下：


    .. math::

        \log \pi_{\theta}(a|s) = -\frac{1}{2}\left(\sum_{i=1}^k \left(\frac{(a_i - \mu_i)^2}{\sigma_i^2} + 2 \log \sigma_i \right) + k \log 2\pi \right).

.. _`分类`: https://en.wikipedia.org/wiki/Categorical_distribution
.. _`高斯`: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
.. _`softmax`: https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax
.. _`tf.distributions.Categorical`: https://www.tensorflow.org/api_docs/python/tf/distributions/Categorical
.. _`tf.multinomial`: https://www.tensorflow.org/api_docs/python/tf/multinomial
.. _`tf.random_normal`: https://www.tensorflow.org/api_docs/python/tf/random_normal
.. _`tf.distributions.Normal`: https://www.tensorflow.org/api_docs/python/tf/distributions/Normal

轨迹（Trajectories）
--------------------

轨迹 :math:`\tau` 指的是环境世界中的一系列的状态和动作。

.. math::

    \tau = (s_0, a_0, s_1, a_1, ...).

第一个状态 :math:`s_0`，是从 **开始状态分布** 中随机采样的，计为 :math:`\rho_0` :

.. math::

    s_0 \sim \rho_0(\cdot).

转态转换（从时间 :math:`t` 的状态 :math:`s_t` 到到时间 :math:`t+1` 的状态 :math:`s_{t+1}` 会发生什么），是由环境的自然规律确定的，并且只依赖于时间 :math:`t` 的动作 :math:`a_t`。它要么是确定性的：

.. math::

    s_{t+1} = f(s_t, a_t)

要么是随机的（当有不确定disturbance输入系统时，时间 :math:`t+1` 的状态还取决于disturbance）：

.. math::

    s_{t+1} \sim P(\cdot|s_t, a_t).

智能体的动作由策略确定。

.. admonition:: 你应该知道

    轨迹（**trajectories**）常常也被称作 **episodes** 或者 **rollouts**。


奖励和回报 (Reward and Return)
-----------------------------

在强化学习中，奖励函数 :math:`R` 非常重要。它由当前时间的状态、当前时间已经执行的动作和下一时间的状态共同决定。

.. math::

    r_t = R(s_t, a_t, s_{t+1})

这个公式常被简化为只依赖当前时间的状态 :math:`r_t = R(s_t)`，或者依赖状态-动作对 :math:`r_t = R(s_t,a_t)`。

智能体的目标是最大化轨迹的累计奖励，这意味着很多事情。我们会把所有的情况表示为 :math:`R(\tau)`，至于具体表示什么，要么可以很清楚的从上下文看出来，要么并不重要。（因为相同的方程式适用于所有情况。）

一种奖励是 **有限时域未折扣奖励** （finite-horizon undiscounted return），它是在一个固定的步骤窗口内获得的奖励的总和：

.. math::

    R(\tau) = \sum_{t=0}^T r_t.

另一种奖励叫做 :math:`\gamma` **无限时域折扣奖励** （infinite-horizon discounted return），指的是智能体获得的全部奖励之和，但是奖励根据未来获得的时间而逐渐折扣。这个公式包含折扣因子 :math:`\gamma \in (0,1)`:

.. math::

    R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t.

这里为什么要加上一个折扣因子呢？为什么不直接把所有的奖励加在一起？确实如此，但折扣因子在直觉上很有吸引力，在数学上也很方便。可以从两个角度来解释： 直观上讲，现在的现金比未来的现金要好；从数学角度讲，无限多个奖励的和很可能 `不收敛`_ 到一个有限值，处理这样一个不收敛的等职是比较困难的。有了衰减因子和合理的约束条件，无限求和就会收敛。

.. admonition:: 你应该知道

    这两个公式从形式上看起来差距很大，事实上我们经常会混用。比如说，我们经常建立算法来优化未折扣的奖励，但在估计 **值函数** 时使用折扣因子。   

.. _`不收敛`: https://en.wikipedia.org/wiki/Convergent_series

强化学习问题（The RL Problem）
----------------------------

无论选择哪种方式衡量收益（无论是无限时域折扣，还是有限时域未折扣），无论选择哪种策略，强化学习的目标都是选择一种策略，当代理根据这个策略行动的时候能最大化 **期望奖励** （expected return）。

讨论期望收益之前，我们先讨论下轨迹的概率分布。

我们假设环境转换和策略都是随机的。这种情况下， :math:`T` 步的轨迹是：

.. math::

    P(\tau|\pi) = \rho_0 (s_0) \prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t) \pi(a_t | s_t).

期望收益计为 :math:`J(\pi)`

.. math::

    J(\pi) = \int_{\tau} P(\tau|\pi) R(\tau) = \underE{\tau\sim \pi}{R(\tau)}.

强化学习中的核心优化问题可以表示为：

.. math::

    \pi^* = \arg \max_{\pi} J(\pi),


:math:`\pi^*` 是 **最优策略** （optimal policy）。

值函数（Value Functions）
-----------------------

知道一个状态或者一对状态-行动(state-action pair)的 **价值** 很有用。这里的价值指的是，如果你从某一个状态或者状态-行动对开始，一直按照某个策略运行下去最终获得的期望回报。几乎所有的强化学习算法，都在用 **值函数** （Value funtion）。

这里介绍四种主要函数：

1. **on-policy值函数** ： :math:`V^{\pi}(s)`，从某一个状态 :math:`s` 开始，之后每一步行动都按照策略 :math:`\pi` 执行
    .. math::
        
        V^{\pi}(s) = \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s\right.}

2. **on-policy动作-值函数** ： :math:`Q^{\pi}(s,a)`,从某一个状态 :math:`s` 开始，先执行任意一个动作 :math:`a` （有可能不是按照策略得到的动作），之后每一步都按照策略 :math:`\pi` 执行：

    .. math::
        
        Q^{\pi}(s,a) = \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s, a_0 = a\right.}


3. **最优值函数**： :math:`V^*(s)`，从某一个状态 :math:`s` 开始，之后每一步都按照 *最优策略*  :math:`\pi` 执行

    .. math::

        V^*(s) = \max_{\pi} \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s\right.}

4.  **最优动作-值函数** ： :math:`Q^*(s,a)` ，从某一个状态 :math:`s` 开始，先执行任意一个动作 :math:`a` （有可能不是按照策略走的），之后每一步都按照 *最优策略* 执行 :math:`\pi`   

    .. math::

        Q^*(s,a) = \max_{\pi} \underE{\tau \sim \pi}{R(\tau)\left| s_0 = s, a_0 = a\right.}

.. admonition:: 你应该知道

    当我们讨论 **值函数** 的时候，如果我们没有提到时间依赖问题，那就意味着这是 **无限时域折扣累计奖励**。 **有限时域无折扣奖励** 需要传入时间作为参数，你知道为什么吗？ 提示：时间到了会发生什么？

.. admonition:: 你应该知道

    值函数和动作-值函数两者之间有关键的联系：

    .. math::

        V^{\pi}(s) = \underE{a\sim \pi}{Q^{\pi}(s,a)},

    以及：

    .. math::

        V^*(s) = \max_a Q^* (s,a).

    这些关系直接来自刚刚给出的定义，你能尝试证明吗？



最优 Q 函数和最优动作（The Optimal Q-Function and the Optimal Action）
-------------------------------------------------------------------

最优动作-值函数 :math:`Q^*(s,a)` 和被最优策略选中的动作有重要的联系。从定义上讲， :math:`Q^*(s,a)` 指的是从一个状态 :math:`s` 开始，任意执行一个动作 :math:`a` ，然后一直按照最优策略执行下去所获得的回报。 

最优策略 :math:`s` 会选择从状态 :math:`s` 开始能够最大化期望回报的动作。所以如果我们有了 :math:`Q^*` ，就可以通过下面的公式直接获得最优动作： :math:`a^*(s)` ：

.. math::

    a^*(s) = \arg \max_a Q^* (s,a).

注意：可能会有多个动作能够最大化 :math:`Q^*(s,a)`，这种情况下，它们都是最优动作，最优策略可能会从中随机选择一个。但是总会存在一个最优策略每一步选择动作的时候是确定的。

贝尔曼方程（Bellman Equations）
------------------------------

全部四个值函数都遵守自一致性的方程叫做 **贝尔曼方程**，贝尔曼方程的基本思想是：

    起始点的值等于当前点的值和接下来到达的状态的值之和。
    
on-policy值函数的贝尔曼方程：

.. math::
    :nowrap:

    \begin{align*}
    V^{\pi}(s) &= \underE{a \sim \pi \\ s'\sim P}{r(s,a) + \gamma V^{\pi}(s')}, \\
    Q^{\pi}(s,a) &= \underE{s'\sim P}{r(s,a) + \gamma \underE{a'\sim \pi}{Q^{\pi}(s',a')}},
    \end{align*}

:math:`s' \sim P` 是 :math:`s' \sim P(\cdot |s,a)` 的简写, 表明下一个时间的状态 :math:`s'` 是按照转换规则从环境中抽样得到的; :math:`a \sim \pi` 是 :math:`a \sim \pi(\cdot|s)` 的简写; and :math:`a' \sim \pi` 是 :math:`a' \sim \pi(\cdot|s')` 的简写. 

最优值函数的贝尔曼方程是：

.. math::
    :nowrap:

    \begin{align*}
    V^*(s) &= \max_a \underE{s'\sim P}{r(s,a) + \gamma V^*(s')}, \\
    Q^*(s,a) &= \underE{s'\sim P}{r(s,a) + \gamma \max_{a'} Q^*(s',a')}.
    \end{align*}

on-policy值函数和最优值函数的贝尔曼方程最大的区别是是否在动作中去 :math:`\max` 。这表明智能体在选择下一步动作时，为了做出最优动作，他必须选择能获得最大值的动作。

.. admonition:: 你应该知道

    贝尔曼算子（Bellman backup）会在强化学习中经常出现。对于一个状态或一个状态-动作对，贝尔曼算子是贝尔曼方程的右边： 奖励加上一个值。
    
优势函数（Advantage Functions）
------------------------------

强化学习中，有些时候我们不需要描述一个动作的绝对好坏，而只需要知道它相对于平均水平的优势。也就是说，我们只想知道一个行动的相对的 **优势** 。这就是优势函数（advantage function）的概念。

一个策略 :math:`\pi` 的优势函数，描述的是它在状态 :math:`s` 下采取行为 :math:`a` 比随机选择一个动作好多少（假设之后一直按照策略 :math:`\pi` 选中动作）。从数学角度，优势函数的定义为：

.. math::

    A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s).

.. admonition:: 你应该知道

    我们之后会继续谈论优势函数，它对于策略梯度方法非常重要。

数学模型（可选）
====================

我们已经非正式地讨论了智能体的环境，但是如果你深入研究，可能会发现这样的标准数学形式：**马尔科夫决策过程** (Markov Decision Processes, MDPs)。MDP是一个5元组 :math:`\langle S, A, R, P, \rho_0 \rangle`，其中

* :math:`S` 是所有有效状态的集合,
* :math:`A` 是所有有效动作的集合,
* :math:`R : S \times A \times S \to \mathbb{R}` 是奖励函数，其中 :math:`r_t = R(s_t, a_t, s_{t+1})`,
* :math:`P : S \times A \to \mathcal{P}(S)` 是转态转移概率函数，其中 :math:`P(s'|s,a)` 是在状态  :math:`s` 下 采取动作 :math:`a` 转移到状态 :math:`s'` 的概率。 
* :math:`\rho_0` 是开始状态的分布。

马尔科夫决策过程指的是服从 `马尔科夫性`_ 的系统： 状态转移只依赖与最近的状态和行动，而不依赖之前的历史数据。

.. _`马尔科夫性`: https://en.wikipedia.org/wiki/Markov_property

