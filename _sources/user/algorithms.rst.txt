==========
核心算法及其实现
==========

.. contents:: 目录

包括哪些算法
===============

下面的算法已经在 Spinning Up 中实现：

- `Vanilla Policy Gradient`_ (VPG)
- `Trust Region Policy Optimization`_ (TRPO)
- `Proximal Policy Optimization`_ (PPO)
- `Deep Deterministic Policy Gradient`_ (DDPG)
- `Twin Delayed DDPG`_ (TD3)
- `Soft Actor-Critic`_ (SAC)

这些算法全部以 `多层感知机`_ actor-critics 的方式实现，从而适用于全观察、不基于图像的强化学习环境，例如 `Gym Mujoco`_ 环境。


.. _`Gym Mujoco`: https://gym.openai.com/envs/#mujoco
.. _`Vanilla Policy Gradient`: ../algorithms/vpg.html
.. _`Trust Region Policy Optimization`: ../algorithms/trpo.html
.. _`Proximal Policy Optimization`: ../algorithms/ppo.html
.. _`Deep Deterministic Policy Gradient`: ../algorithms/ddpg.html
.. _`Twin Delayed DDPG`: ../algorithms/td3.html
.. _`Soft Actor-Critic`: ../algorithms/sac.html
.. _`多层感知机`: https://en.wikipedia.org/wiki/Multilayer_perceptron


为什么介绍这些算法？
=====================

我们在这个项目中选取了能够呈现强化学习近些年发展历程的核心算法。目前，在 **可靠性** (stability)和 **采样效率** (sample efficiency)这两个因素上表现最优的策略学习算法是 PPO 和 SAC。从这些算法的设计和实际应用中，可以看出可靠性和采样效率两者的权衡。


同策略（On-Policy）算法
------------------------

Vanilla Policy Gradient(VPG) 是深度强化学习领域最基础也是入门级的算法，发表时间远早于深度强化学习。VPG 算法的核心思想可以追溯到上世纪 80 年代末、90年代初。在那之后，TRPO（2015）和 PPO(2017) 等更好的算法才相继诞生。

上述系列工作都是基于不使用历史数据的同策略，因此在采样效率上表现相对较差。但这也是有原因的：它们直接优化我们关心的目标 —— 策略表现。这个系列的算法都是用采样效率换取可靠性，之后提出的算法，从 VPG 到TRPO 再到 PPO，都是在不断弥补采样效率方面的不足。

异策略（Off-Policy）算法
-------------------------

DDPG 是一个和 VPG 同样重要的算法，尽管它的提出时间较晚。确定策略梯度（Deterministic Policy Gradients，DPG）理论是在 2014 年提出的，是 DDPG 算法的基础。DDPG 算法和 Q-learning 算法很相似，都是同时学习 Q 函数和策略并通过更新相互提高。

DDPG 和 Q-Learning 属于 *异策略* 算法，他们通过对贝尔曼方程（Bellman’s equations,也称动态规划方程）的优化，实现对历史数据的有效利用。

但问题是，满足贝尔曼方程并不能保证一定有很好的策略性能。从经验上讲，满足贝尔曼方程可以有不错的性能、很好的采样效率,但也由于没有这种必然性的保证，这类算法没有那么稳定。基于 DDPG的后续工作 TD3 和 SAC 提出了很多新的方案来缓解这些问题。

代码格式
===========

Spinning Up 项目的算法都按照固定的模板来实现。每个算法由两个文件组成：

* 算法文件，主要是算法的核心逻辑
* 核心文件，包括各种运行算法所需的工具类。

算法文件
------------------

算法文件最开始是经验存储类(experience buffer)的定义，作用是存储智能体和环境的互动信息。

接下来有一个运行算法，以及以下算法：
    
    1) Logger 输出设定

    2) 随机种子的设定
    
    3) 环境实例化
    
    4) 为计算图创建 placeholder
    
    5) 通过 *actor-critic* 函数传递算法函数

    6) 实例化经验缓存
    
    7) 损失函数和一些其他的函数
    
    8) 构建训练 ops
    
    9) 构建 TF Session 并初始化参数
    
    10) 通过 logger 保存模型
    
    11) 定义运行算法主循环需要的函数（例如核心更新函数，获取行动函数，测试智能体函数等，取决于具体的算法）
    
    12) 运行算法主循环
    
        a) 让智能体在环境中开始运行
    
        b) 根据算法的主要方程式，周期性更新参数
    
        c) 打印核心性能数据并保存智能体

最后是从命令行读入设置的代码(ArgumentParser)。

核心文件
-------------

核心文件并没有像算法文件那样严格遵守模板，但也有一些相似的结构。

    1) 构建 placeholder 的函数

    2) *actor-critic* 有关函数

    3) 其他有用的函数

    4) 与算法兼容的多层感知机 actor-critic 实现，策略和值函数(value function)都是通过简单的多层感知机来表示


