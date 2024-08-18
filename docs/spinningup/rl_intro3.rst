====================================
第三部分：策略优化介绍
====================================

.. contents:: Table of Contents
    :depth: 2


在这个部分，我们会讨论策略优化算法的数学基础，同时提供样例代码。我们会包括策略优化的以下三个部分

* **最简单的等式**描述策略性能对于策略参数的梯度，
* 一个让我们可以**舍弃无用项**的公式，
* 一个让我们可以**添加有用参数**的公式。

最后，我们会把结果放在一起，然后描述基于优势函数的策略梯度的表达式： 我们在 `Vanilla Policy Gradient`_ 的实现中使用的版本。

.. _`最简单的等式`: ../spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
.. _`舍弃无用项`: ../spinningup/rl_intro3.html#don-t-let-the-past-distract-you
.. _`增加有用项`: ../spinningup/rl_intro3.html#baselines-in-policy-gradients
.. _`Vanilla Policy Gradient`: ../algorithms/vpg.html

推导最简单的策略梯度（Deriving the Simplest Policy Gradient）
=========================================================

我们考虑一种基于随机参数化的策略： :math:`\pi_{\theta}` 。我们的目的是最大化期望回报 :math:`J(\pi_{\theta}) = \mathop{\mathbb{E}}\limits_{\tau \sim \pi_{\theta}}[R(\tau)]` 。为了公式推导，我们假定 :math:`R(\tau)` 是 `有限时域无折扣回报`，但是对于无限时域折扣回报来说也是一样的。


.. _`有限时域无折扣回报`: ../spinningup/rl_intro.html#reward-and-return

我们想要通过梯度下降（gradient ascent）来优化策略，例如

.. math::

    \theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}.

策略性能的梯度 :math:`\nabla_{\theta} J(\pi_{\theta})` ，通常被称为 **策略梯度** (policy gradient) ，以这种方式优化策略的算法被称为 **策略梯度算法** (policy gradient algorithms)。（比如说 Vanilla Policy Gradient 和 TRPO。PPO 也被称为策略梯度算法，尽管这样说不是很准确。）

为了在实际中使用这个算法, 我们需要能够在数值计算中使用的表达式. 这涉及到两个步骤: 1) 推导策略性能的梯度的解析式, 它的形式是一个期望值 2) 计算期望值的样本估计，可以用有限数量的智能体和环境交互的数据来计算。


在这个小节中，我们介绍这个表达式最简单的形式. 这之后的小节中, 我们将展示如何改进最简单的形式，以获得我们在标准策略梯度算法实现中实际使用的版本。

我们首先列出一些对推导梯度解析式有用的等式。

**1. 轨迹的概率（Probability of a Trajectory）。** 已知动作（actions）来自于随机策略 :math:`\pi_{\theta}` 的一个轨迹（trajectory等价于episode或rollout） :math:`\tau = (s_0, a_0, ..., s_{T+1})` 发生的概率是:

.. math::

    P(\tau|\theta) = \rho_0 (s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t |s_t).

:math:`s_0 ~ \rho_0 (s)` 是初始状态分布； :math:`s_{t+1} ~ P(s_{t+1}|s_t,a_t)` 是状态转移概率； :math:`\pi_{\theta}(a_t |s_t)` 是随机策略。


**2. 对数求导技巧(The Log-Derivative Trick).** 对数求导的技巧是基于微积分中的简单求导法则: :math:`\log x` 对 :math:`x` 求导的结果是 :math:`1/x` 。当重新排列并结合链式法则时，我们得到:

.. math::

    \nabla_{\theta} P(\tau | \theta) = P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta).


**3. 轨迹发生概率的对数（Log-Probability of a Trajectory）.** 一个轨迹发生概率的对数为

.. math::

    \log P(\tau|\theta) = \log \rho_0 (s_0) + \sum_{t=0}^{T} \bigg( \log P(s_{t+1}|s_t, a_t)  + \log \pi_{\theta}(a_t |s_t)\bigg).


**4. 环境函数的梯度（Gradients of Environment Functions）.** 环境和 :math:`\theta` 无关,  :math:`\rho_0(s_0)`, :math:`P(s_{t+1}|s_t, a_t)`, 和 :math:`R(\tau)` 对 :math:`\theta` 的梯度为0.

**5. 轨迹发生概率的对数的梯度（Grad-Log-Prob of a Trajectory）.** 轨迹发生概率的对数的梯度可以表示为：

.. math::

    \nabla_{\theta} \log P(\tau | \theta) &= \cancel{\nabla_{\theta} \log \rho_0 (s_0)} + \sum_{t=0}^{T} \bigg( \cancel{\nabla_{\theta} \log P(s_{t+1}|s_t, a_t)}  + \nabla_{\theta} \log \pi_{\theta}(a_t |s_t)\bigg) \\
    &= \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t).


把上面五个规则代入到一起我们可以推导出:

.. admonition:: 基本的策略梯度的推导

    .. math::
        :nowrap:

        \begin{align*}
        \nabla_{\theta} J(\pi_{\theta}) &= \nabla_{\theta} \underE{\tau \sim \pi_{\theta}}{R(\tau)} & \\
        &= \nabla_{\theta} \int_{\tau} P(\tau|\theta) R(\tau) & \text{展开期望} \\
        &= \int_{\tau} \nabla_{\theta} P(\tau|\theta) R(\tau) & \text{把梯度算子代入到积分中} \\
        &= \int_{\tau} P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) R(\tau) & \text{对数求导技巧(The Log-Derivative Trick)} \\
        &= \underE{\tau \sim \pi_{\theta}}{\nabla_{\theta} \log P(\tau|\theta) R(\tau)} & \text{再转化为一个期望的形式} \\
        \therefore \nabla_{\theta} J(\pi_{\theta}) &= \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)} & \text{轨迹概率对数梯度的表达式}
        \end{align*}

这是一个期望，我们可以使用样本均值去估计它. 如果我们收集了一个轨迹集合 :math:`\mathcal{D} = \{\tau_i\}_{i=1,...,N}` 式子中的轨迹是通过让智能体在环境中使用策略 :math:`\pi_{\theta}` 生成动作来指导运行获得的, 策略梯度则能够使用下式估计：

.. math::

    \hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau),

式中 :math:`|\mathcal{D}|` 集合 :math:`\mathcal{D}` 中轨迹的数量(这里定义为, :math:`N`).

最后一个表达式是我们想要的可以进行计算的最简单版本。 假设我们已经用一种可以计算 :math:`\nabla_{\theta} \log \pi_{\theta}(a|s)` 的方式来表示我们的策略且我们能够运行在环境中运行策略去收集轨迹中的数据集, 那么我们就可以计算策略梯度并执行更新步骤.

实现最简单的策略梯度（Implementing the Simplest Policy Gradient）
==============================================================

在 ``spinup/examples/pytorch/pg_math/1_simple_pg.py`` 中我们给出了这个简单版本的策略梯度算法的简短PyTorch实现。 (它也能在 `github <https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py>`_ 查看。) 它只有128行，所以我们强烈建议深入阅读。虽然我们不会在这里讨论全部代码，但我们将重点介绍和解释一些重要的部分。

.. admonition:: 你应该知道

    这部分先前写了一个Tensorflow的例子，老的Tensorflow部分能够在 `这里 <https://spinningup.openai.com/en/latest/spinningup/extra_tf_pg_implementation.html#implementing-the-simplest-policy-gradient>`_ 查看。


**1. 创建一个策略网络。** 

.. code-block:: python
    :linenos:
    :lineno-start: 30

    # 创建策略网络的核心
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # 创建一个函数去计算动作分布
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # 创建通过选择的函数（输出是一个从策略中采样的动作）
    def get_action(obs):
        return get_policy(obs).sample().item()

本模块构建了使用前馈神经网络分类策略的模块和函数。 (查看第一部分的 `Stochastic Policies`_ 章节进行回顾。)  ``logits_net`` 模块的输出可以被用来构建概率的对数和动作的概率, ``get_action`` 函数基于 ``logits`` 计算的概率对动作进行采样。
（注意： ``get_action`` 函数假设仅有一个 ``obs`` 被提供，因此仅有一个整数的动作输出，这就是为什么使用了 ``.item()`` ,使用这个能够 `从张量中提取一个元素 <https://pytorch.org/docs/stable/tensors.html#torch.Tensor.item>`_ 。）

.. _`Stochastic Policies`: ../spinningup/rl_intro.html#stochastic-policies

在这个例子中，大量的工作都被35行的 ``Categorical`` 对象完成了。这是一个PyTorch版本的 ``Distribution`` 对象，它封装了一些域概率分布相关的数学函数。特别是，有一个可以从分布中进行采样的方法（这个方法在第40行中被使用）和一个计算给定样本对数概率的方法（这个方法在之后会提到）。由于PyTorch的分布对强化学习来说真的很有用，查看它们的 ``文档 <https://pytorch.org/docs/stable/distributions.html>``_ ，了解它们是如何工作的。

.. admonition:: 你应该知道

    温馨提示！当我们提到categorical分布有一“logits”，意思是每一个结果的概率都是logits的Softmax函数的输出。也就是说，在一个包含logits :math: x_j 的categorical分布动作 :math:`j` 的概率是：

    .. math::

        p_j = \frac{{\rm{exp}}(x_j)}{\sum_i{\rm{exp}}(x_i}}

**2. 创建一个损失函数。**

.. code-block:: python
    :linenos:
    :lineno-start: 42

    # 构造损失函数，输出正确的数据，输出策略梯度
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()


在本节中，我们为策略梯度算法构建了一个“损失”函数。 当输入正确的数据时，该损失的梯度等于策略梯度。 正确的数据是指根据当前策略操作时收集的一组(状态、动作、权重)元组，其中状态-动作对的权重是它所属轨迹（trajectory，episode，or rollout）的返回值。(尽管我们将在后面的小节中展示，您可以为权重插入其他值，这些值也可以正常工作。)


.. admonition:: 你应该知道
    
    尽管我们将其描述为损失函数，但它并**不**是监督学习中典型意义上的损失函数。它与标准损失函数有两个主要区别。

    **1. 数据分布取决于参数。** 损失函数通常定义在一个与我们要优化的参数无关的固定数据分布上。这里的情况并非如此，数据必须取自最近的策略。 

    **2. 它并不衡量性能。** 损失函数通常评估我们关心的性能指标。这里，我们关心的是预期收益, :math:`J(\pi_{\theta})`, 但是我们的“损失”函数根本不接近这个预期收益，即使是它的期望也不接近。这个“损失”函数只对我们有用，因为当对当前参数进行评估时，使用当前参数生成的数据，它具有负的性能梯度。 

    但在第一步梯度下降之后，与性能就没有任何联系了。这意味着最小化这个“损失”函数，对于给定的一批数据，并不能保证提高预期回报。你可以把这笔损失减小到 :math:`-\infty` 同时策略的性能非常差；事实上，它通常是这样的。有时，深度强化学习研究人员可能会将这种结果描述为策略对一批数据的“过拟合”。这是未来方便理解，但不同于监督学习的“过拟合”，因为这里不涉及泛化误差。

   我们之所以提出这一点，是因为机器学习从业者通常会在训练期间将损失函数解释为有用的信号——“如果损失下降，一切都很好。”在策略梯度中，这种直觉是错误的，你应该只关心平均收益。损失函数没有任何意义。




.. admonition:: 你应该知道
    
    这里使用的这个方法调用 ``log_prob`` PyTorch 中``Categorical`` 对象的 ``log_prob`` 方法创建一个 ``logp`` 张量。如果要使用其他分布可能需要一些修改。

    例如，如果你正在使用正态分布（对角高斯策略），调用 ``policy.log_prob(act)`` 的输出将为您提供一个张量，其中包含每个向量值动作的每个元素的单独的对数概率。也就是说，当你需要的是一个形状为(batch，)张量创建强化学习的损失的时候，你输入一个形状为``(batch, act_dim)``的张量，得到一个形状``(batch, act_dim``的张量。在这种情况下，你将动作元素的对数概率相加，得到动作的对数概率。也就是说，你讲计算:

    .. code-block::

    logp = get_policy(obs).log_prob(act).sum(axis=-1)

**3. 运行训练的一代（epoch）。**

.. code-block:: python
    :linenos:
    :lineno-start: 50

    # 训练策略
    def train_one_epoch():
        # 创建一些空的列表用于存储日志
        batch_obs = []          # 观测
        batch_acts = []         # 动作
        batch_weights = []      # 策略梯度中 R(tau) 的权重
        batch_rets = []         # 测量轨迹的回报值
        batch_lens = []         # 测量轨迹的长度
    
        # 重设与轨迹相关的变量
        obs = env.reset()       # 来自于起始分布的第一个观测
        done = False            # 环境中轨迹结束的信号
        ep_rews = []            # 整个轨迹累计的奖励的列表
    
        # 渲染每代的第一个轨迹
        finished_rendering_this_epoch = False
    
        # 通过在当前测量的环境中动作来收集经历
        while True:
    
            # 渲染
            if (not finished_rendering_this_epoch) and render:
                env.render()
    
            # 保存观测值
            batch_obs.append(obs.copy())
    
            # 在环境中执行动作（状态转移）
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)
    
            # 保存动作和奖励
            batch_acts.append(act)
            ep_rews.append(rew)
    
            if done:
                # 如果轨迹结果的话，记录和轨迹相关的一些信息
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
    
                # 每个logprob(a|s) 的权重是 R(tau)
                batch_weights += [ep_ret] * ep_len
    
                # 重设与轨迹相关的变量
                obs, done, ep_rews = env.reset(), False, []
    
                # 这个代不会再渲染了
                finished_rendering_this_epoch = True
    
                # 结束循环，如果我们有足够的经历
                if len(batch_obs) > batch_size:
                    break
    
        # 执行一步测量更新的步骤
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

``train_one_epoch()`` 函数运行策略梯度的一个“代”, 我们的定义是 

1) 经验收集步骤 (67-102行), 智能体在环境中使用最近的策略执行一定数量的轨迹，然后是 

2) 单次策略梯度更新的步骤 (99-105行). 

算法的主循环只是重复调用 ``train_one_epoch()``. 

.. admonition:: 你应该知道
    
    如果您还不熟悉PyTorch中的优化，请观察执行一个梯度下降步骤的模式，如第104-111行所示。首先，清除梯度缓存。然后，计算损失函数。然后，计算损失函数的反向传递;这会将新的梯度累积到梯度缓冲区中。最后，使用优化器执行一步。

概率的对数的梯度的期望的引理（Expected Grad-Log-Prob Lemma）
=========================================================

在本节中，我们将推导出一个中间结果，它在整个策略梯度理论中被广泛使用。我们把它叫做 Expected Grad-Log-Prob (EGLP) 引理. [1]_

**EGLP Lemma.** 假设 :math:`P_{\theta}` 是一个随机变量 :math:`x` 的参数化概率分布。 则: 

.. math::

    \underE{x \sim P_{\theta}}{\nabla_{\theta} \log P_{\theta}(x)} = 0.

.. admonition:: 证明

    回想一下，所有的概率分布都是**标准化**的:

    .. math::

        \int_x P_{\theta}(x) = 1.

    对归一化条件的等式两侧取梯度:

    .. math::

        \nabla_{\theta} \int_x P_{\theta}(x) = \nabla_{\theta} 1 = 0.

    用对数导数的技巧得到:

    .. math::

        0 &= \nabla_{\theta} \int_x P_{\theta}(x) \\
        &= \int_x \nabla_{\theta} P_{\theta}(x) \\
        &= \int_x P_{\theta}(x) \nabla_{\theta} \log P_{\theta}(x) \\
        \therefore 0 &= \underE{x \sim P_{\theta}}{\nabla_{\theta} \log P_{\theta}(x)}.

.. [1] 本文的作者没有意识到这个引理在任何文献中都有一个标准的名称。但考虑到它出现的频率，为方便参考，给它起个名字似乎是很值得的。

不要让过去分散你的注意力
===============================

检查我们最近的策略梯度的表达式:

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)}.
 
采用这种梯度走一步，每个动作的对数概率就会与 :math:`R(\tau)` 成比例增加，即 *所有获得的奖励总和*。但这没有多大意义。

智能体确实应该只在“结果”的基础上强化动作。在采取动作之前获得的奖励与该行动有多好无关:只有“之后”才会获得奖励。

这种直觉在数学中也有体现，我们可以证明策略梯度也可以用

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})}.

在这种形式中，行动只会基于采取动作后获得的奖励而得到强化。

我们称这种形式为 "reward-to-go policy gradient," 因为是轨迹上某一点之后的奖励总和，

.. math::

    \hat{R}_t \doteq \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}),

从这一点之后被称为**reward-to-go**,这个策略梯度表达式依赖于来自状态-行动对的奖励。

.. admonition:: You Should Know

    **But how is this better?** A key problem with policy gradients is how many sample trajectories are needed to get a low-variance sample estimate for them. The formula we started with included terms for reinforcing actions proportional to past rewards, all of which had zero mean, but nonzero variance: as a result, they would just add noise to sample estimates of the policy gradient. By removing them, we reduce the number of sample trajectories needed.

An (optional) proof of this claim can be found `here`_, and it ultimately depends on the EGLP lemma.

.. _`here`: ../spinningup/extra_pg_proof1.html

Implementing Reward-to-Go Policy Gradient
=========================================

We give a short Tensorflow implementation of the reward-to-go policy gradient in ``spinup/examples/pg_math/2_rtg_pg.py``. (It can also be viewed `on github <https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/2_rtg_pg.py>`_.) 

The only thing that has changed from ``1_simple_pg.py`` is that we now use different weights in the loss function. The code modification is very slight: we add a new function, and change two other lines. The new function is:

.. code-block:: python
    :linenos:
    :lineno-start: 12

    def reward_to_go(rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs


And then we tweak the old L86-87 from:

.. code-block:: python
    :linenos:
    :lineno-start: 86

                    # the weight for each logprob(a|s) is R(tau)
                    batch_weights += [ep_ret] * ep_len

to:

.. code-block:: python
    :linenos:
    :lineno-start: 93

                    # the weight for each logprob(a_t|s_t) is reward-to-go from t
                    batch_weights += list(reward_to_go(ep_rews))



Baselines in Policy Gradients
=============================

An immediate consequence of the EGLP lemma is that for any function :math:`b` which only depends on state,

.. math::

    \underE{a_t \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) b(s_t)} = 0.

This allows us to add or subtract any number of terms like this from our expression for the policy gradient, without changing it in expectation:

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \left(\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)\right)}.

Any function :math:`b` used in this way is called a **baseline**. 

The most common choice of baseline is the `on-policy value function`_ :math:`V^{\pi}(s_t)`. Recall that this is the average return an agent gets if it starts in state :math:`s_t` and then acts according to policy :math:`\pi` for the rest of its life. 

Empirically, the choice :math:`b(s_t) = V^{\pi}(s_t)` has the desirable effect of reducing variance in the sample estimate for the policy gradient. This results in faster and more stable policy learning. It is also appealing from a conceptual angle: it encodes the intuition that if an agent gets what it expected, it should "feel" neutral about it.

.. admonition:: You Should Know

    In practice, :math:`V^{\pi}(s_t)` cannot be computed exactly, so it has to be approximated. This is usually done with a neural network, :math:`V_{\phi}(s_t)`, which is updated concurrently with the policy (so that the value network always approximates the value function of the most recent policy).

    The simplest method for learning :math:`V_{\phi}`, used in most implementations of policy optimization algorithms (including VPG, TRPO, PPO, and A2C), is to minimize a mean-squared-error objective:

    .. math:: \phi_k = \arg \min_{\phi} \underE{s_t, \hat{R}_t \sim \pi_k}{\left( V_{\phi}(s_t) - \hat{R}_t \right)^2},

    | 
    where :math:`\pi_k` is the policy at epoch :math:`k`. This is done with one or more steps of gradient descent, starting from the previous value parameters :math:`\phi_{k-1}`. 


Other Forms of the Policy Gradient
==================================

What we have seen so far is that the policy gradient has the general form

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Phi_t},

where :math:`\Phi_t` could be any of

.. math:: \Phi_t &= R(\tau), 

or

.. math:: \Phi_t &= \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}), 

or 

.. math:: \Phi_t &= \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t).

All of these choices lead to the same expected value for the policy gradient, despite having different variances. It turns out that there are two more valid choices of weights :math:`\Phi_t` which are important to know.

**1. On-Policy Action-Value Function.** The choice

.. math:: \Phi_t = Q^{\pi_{\theta}}(s_t, a_t)

is also valid. See `this page`_ for an (optional) proof of this claim.

**2. The Advantage Function.** Recall that the `advantage of an action`_, defined by :math:`A^{\pi}(s_t,a_t) = Q^{\pi}(s_t,a_t) - V^{\pi}(s_t)`,  describes how much better or worse it is than other actions on average (relative to the current policy). This choice,

.. math:: \Phi_t = A^{\pi_{\theta}}(s_t, a_t)

is also valid. The proof is that it's equivalent to using :math:`\Phi_t = Q^{\pi_{\theta}}(s_t, a_t)` and then using a value function baseline, which we are always free to do.

.. admonition:: You Should Know

    The formulation of policy gradients with advantage functions is extremely common, and there are many different ways of estimating the advantage function used by different algorithms.

.. admonition:: You Should Know

    For a more detailed treatment of this topic, you should read the paper on `Generalized Advantage Estimation`_ (GAE), which goes into depth about different choices of :math:`\Phi_t` in the background sections.

    That paper then goes on to describe GAE, a method for approximating the advantage function in policy optimization algorithms which enjoys widespread use. For instance, Spinning Up's implementations of VPG, TRPO, and PPO make use of it. As a result, we strongly advise you to study it.


Recap
=====

In this chapter, we described the basic theory of policy gradient methods and connected some of the early results to code examples. The interested student should continue from here by studying how the later results (value function baselines and the advantage formulation of policy gradients) translate into Spinning Up's implementation of `Vanilla Policy Gradient`_.

.. _`on-policy value function`: ../spinningup/rl_intro.html#value-functions
.. _`advantage of an action`: ../spinningup/rl_intro.html#advantage-functions
.. _`this page`: ../spinningup/extra_pg_proof2.html
.. _`Generalized Advantage Estimation`: https://arxiv.org/abs/1506.02438
.. _`Vanilla Policy Gradient`: ../algorithms/vpg.html

