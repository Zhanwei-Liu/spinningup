====================================
第三部分：策略优化介绍
====================================

.. contents:: Table of Contents
    :depth: 2


在这个部分，我们会讨论策略优化算法的数学基础，同时提供样例代码。我们会包括策略优化的以下三个部分

* 最简单的等式 描述了策略表现对于策略参数的梯度
* 一个让我们可以 舍弃无用项 的公式
* 一个让我们可以 添加有用参数 的公式

最后，我们会把结果放在一起，然后描述策略梯度更有优势的版本： 我们在 `Vanilla Policy Gradient`_ 中使用的版本。

.. _`the simplest equation`: ../spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
.. _`drop useless terms`: ../spinningup/rl_intro3.html#don-t-let-the-past-distract-you
.. _`add useful terms`: ../spinningup/rl_intro3.html#baselines-in-policy-gradients
.. _`Vanilla Policy Gradient`: ../algorithms/vpg.html

最简单的策略梯度求导
=====================================

我们考虑一种基于随机策略： :math:`\pi_{\theta}` 。我们的目的是最小化期望回报 :math:`J(\pi_{\theta}) = \mathop{\mathbb{E}}\limits_{\tau \sim \pi_{\theta}}[R(\tau)]` 。为了计算导数，我们假定 :math:`R(\tau)` 是 `有限时域无衰减回报`，但是对于无限时域衰减回报来说也是一样的。


.. _`有限时域无衰减回报`: ../spinningup/rl_intro.html#reward-and-return

我们想要通过梯度下降来优化策略，例如

.. math::

    \theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}.

策略性能的梯度 :math:`\nabla_{\theta} J(\pi_{\theta})` ，通常被称为 **策略梯度** (policy gradient) ，以这种方式优化策略的算法被称为 **策略梯度算法** (policy gradient algorithms)。（比如说 Vanilla Policy Gradient 和 TRPO。PPO 也被称为策略梯度算法，尽管这样说不是很准确。）

为了在实际中使用这个算法, 我们需要能够在数值计算中使用的表达式. 这涉及到两个步骤: 1) 推导策略性能的梯度的解析式, 它的形式是一个期望值 2) 计算期望值的样本估计，可以用有限数量的代理和环境交互的数据来计算。


在这个小节中，我们介绍这个表达式最简单的形式. 这之后的小节中, 我们将展示如何改进最简单的形式，以获得我们在标准策略梯度算法实现中实际使用的版本。

我们首先列出一些对推导解析梯度有用的等式。

**1. 轨迹的概率（Probability of a Trajectory）。** 行为（actions）来自于已知参数 :math:`\theta` 的随机策略 :math:`\pi_{\theta}` 的一个轨迹（trajectory等价于episode或rollout） :math:`\tau = (s_0, a_0, ..., s_{T+1})` 发生的概率是:

.. math::

    P(\tau|\theta) = P_0 (s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t |s_t).

:math:`s_0 ~ P_0 (s)` 是初始状态分布； :math:`s_{t+1} ~ P(s_{t+1}|s_t,a_t)` 是状态转移概率； :math:`\pi_{\theta}(a_t |s_t)` 是随机策略。




**2. The Log-Derivative Trick.** The log-derivative trick is based on a simple rule from calculus: the derivative of :math:`\log x` with respect to :math:`x` is :math:`1/x`. When rearranged and combined with chain rule, we get:

.. math::

    \nabla_{\theta} P(\tau | \theta) = P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta).


**3. Log-Probability of a Trajectory.** The log-prob of a trajectory is just

.. math::

    \log P(\tau|\theta) = \log \rho_0 (s_0) + \sum_{t=0}^{T} \bigg( \log P(s_{t+1}|s_t, a_t)  + \log \pi_{\theta}(a_t |s_t)\bigg).


**4. Gradients of Environment Functions.** The environment has no dependence on :math:`\theta`, so gradients of :math:`\rho_0(s_0)`, :math:`P(s_{t+1}|s_t, a_t)`, and :math:`R(\tau)` are zero.

**5. Grad-Log-Prob of a Trajectory.** The gradient of the log-prob of a trajectory is thus

.. math::

    \nabla_{\theta} \log P(\tau | \theta) &= \cancel{\nabla_{\theta} \log \rho_0 (s_0)} + \sum_{t=0}^{T} \bigg( \cancel{\nabla_{\theta} \log P(s_{t+1}|s_t, a_t)}  + \nabla_{\theta} \log \pi_{\theta}(a_t |s_t)\bigg) \\
    &= \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t).


Putting it all together, we derive the following:

.. admonition:: Derivation for Basic Policy Gradient

    .. math::
        :nowrap:

        \begin{align*}
        \nabla_{\theta} J(\pi_{\theta}) &= \nabla_{\theta} \underE{\tau \sim \pi_{\theta}}{R(\tau)} & \\
        &= \nabla_{\theta} \int_{\tau} P(\tau|\theta) R(\tau) & \text{Expand expectation} \\
        &= \int_{\tau} \nabla_{\theta} P(\tau|\theta) R(\tau) & \text{Bring gradient under integral} \\
        &= \int_{\tau} P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) R(\tau) & \text{Log-derivative trick} \\
        &= \underE{\tau \sim \pi_{\theta}}{\nabla_{\theta} \log P(\tau|\theta) R(\tau)} & \text{Return to expectation form} \\
        \therefore \nabla_{\theta} J(\pi_{\theta}) &= \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)} & \text{Expression for grad-log-prob}
        \end{align*}

This is an expectation, which means that we can estimate it with a sample mean. If we collect a set of trajectories :math:`\mathcal{D} = \{\tau_i\}_{i=1,...,N}` where each trajectory is obtained by letting the agent act in the environment using the policy :math:`\pi_{\theta}`, the policy gradient can be estimated with

.. math::

    \hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau),

where :math:`|\mathcal{D}|` is the number of trajectories in :math:`\mathcal{D}` (here, :math:`N`).

This last expression is the simplest version of the computable expression we desired. Assuming that we have represented our policy in a way which allows us to calculate :math:`\nabla_{\theta} \log \pi_{\theta}(a|s)`, and if we are able to run the policy in the environment to collect the trajectory dataset, we can compute the policy gradient and take an update step.

Implementing the Simplest Policy Gradient
=========================================

We give a short Tensorflow implementation of this simple version of the policy gradient algorithm in ``spinup/examples/pg_math/1_simple_pg.py``. (It can also be viewed `on github <https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py>`_.) It is only 122 lines long, so we highly recommend reading through it in depth. While we won't go through the entirety of the code here, we'll highlight and explain a few important pieces.

**1. Making the Policy Network.** 

.. code-block:: python
    :linenos:
    :lineno-start: 25

    # make core of policy network
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])

    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)

This block builds a feedforward neural network categorical policy. (See the `Stochastic Policies`_ section in Part 1 for a refresher.) The ``logits`` tensor can be used to construct log-probabilities and probabilities for actions, and the ``actions`` tensor samples actions based on the probabilities implied by ``logits``.

.. _`Stochastic Policies`: ../spinningup/rl_intro.html#stochastic-policies

**2. Making the Loss Function.**

.. code-block:: python
    :linenos:
    :lineno-start: 32

    # make loss function whose gradient, for the right data, is policy gradient
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    action_masks = tf.one_hot(act_ph, n_acts)
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    loss = -tf.reduce_mean(weights_ph * log_probs)


In this block, we build a "loss" function for the policy gradient algorithm. When the right data is plugged in, the gradient of this loss is equal to the policy gradient. The right data means a set of (state, action, weight) tuples collected while acting according to the current policy, where the weight for a state-action pair is the return from the episode to which it belongs. (Although as we will show in later subsections, there are other values you can plug in for the weight which also work correctly.)


.. admonition:: You Should Know
    
    Even though we describe this as a loss function, it is **not** a loss function in the typical sense from supervised learning. There are two main differences from standard loss functions.

    **1. The data distribution depends on the parameters.** A loss function is usually defined on a fixed data distribution which is independent of the parameters we aim to optimize. Not so here, where the data must be sampled on the most recent policy. 

    **2. It doesn't measure performance.** A loss function usually evaluates the performance metric that we care about. Here, we care about expected return, :math:`J(\pi_{\theta})`, but our "loss" function does not approximate this at all, even in expectation. This "loss" function is only useful to us because, when evaluated at the current parameters, with data generated by the current parameters, it has the negative gradient of performance. 

    But after that first step of gradient descent, there is no more connection to performance. This means that minimizing this "loss" function, for a given batch of data, has *no* guarantee whatsoever of improving expected return. You can send this loss to :math:`-\infty` and policy performance could crater; in fact, it usually will. Sometimes a deep RL researcher might describe this outcome as the policy "overfitting" to a batch of data. This is descriptive, but should not be taken literally because it does not refer to generalization error.

    We raise this point because it is common for ML practitioners to interpret a loss function as a useful signal during training---"if the loss goes down, all is well." In policy gradients, this intuition is wrong, and you should only care about average return. The loss function means nothing.




.. admonition:: You Should Know
    
    The approach used here to make the ``log_probs`` tensor---creating an action mask, and using it to select out particular log probabilities---*only* works for categorical policies. It does not work in general. 



**3. Running One Epoch of Training.**

.. code-block:: python
    :linenos:
    :lineno-start: 45

        # for training policy
        def train_one_epoch():
            # make some empty lists for logging.
            batch_obs = []          # for observations
            batch_acts = []         # for actions
            batch_weights = []      # for R(tau) weighting in policy gradient
            batch_rets = []         # for measuring episode returns
            batch_lens = []         # for measuring episode lengths

            # reset episode-specific variables
            obs = env.reset()       # first obs comes from starting distribution
            done = False            # signal from environment that episode is over
            ep_rews = []            # list for rewards accrued throughout ep

            # render first episode of each epoch
            finished_rendering_this_epoch = False

            # collect experience by acting in the environment with current policy
            while True:

                # rendering
                if not(finished_rendering_this_epoch):
                    env.render()

                # save obs
                batch_obs.append(obs.copy())

                # act in the environment
                act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]
                obs, rew, done, _ = env.step(act)

                # save action, reward
                batch_acts.append(act)
                ep_rews.append(rew)

                if done:
                    # if episode is over, record info about episode
                    ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                    batch_rets.append(ep_ret)
                    batch_lens.append(ep_len)

                    # the weight for each logprob(a|s) is R(tau)
                    batch_weights += [ep_ret] * ep_len

                    # reset episode-specific variables
                    obs, done, ep_rews = env.reset(), False, []

                    # won't render again this epoch
                    finished_rendering_this_epoch = True

                    # end experience loop if we have enough of it
                    if len(batch_obs) > batch_size:
                        break

            # take a single policy gradient update step
            batch_loss, _ = sess.run([loss, train_op],
                                     feed_dict={
                                        obs_ph: np.array(batch_obs),
                                        act_ph: np.array(batch_acts),
                                        weights_ph: np.array(batch_weights)
                                     })
            return batch_loss, batch_rets, batch_lens

The ``train_one_epoch()`` function runs one "epoch" of policy gradient, which we define to be 

1) the experience collection step (L62-97), where the agent acts for some number of episodes in the environment using the most recent policy, followed by 

2) a single policy gradient update step (L99-105). 

The main loop of the algorithm just repeatedly calls ``train_one_epoch()``. 


Expected Grad-Log-Prob Lemma
============================

In this subsection, we will derive an intermediate result which is extensively used throughout the theory of policy gradients. We will call it the Expected Grad-Log-Prob (EGLP) lemma. [1]_

**EGLP Lemma.** Suppose that :math:`P_{\theta}` is a parameterized probability distribution over a random variable, :math:`x`. Then: 

.. math::

    \underE{x \sim P_{\theta}}{\nabla_{\theta} \log P_{\theta}(x)} = 0.

.. admonition:: Proof

    Recall that all probability distributions are *normalized*:

    .. math::

        \int_x P_{\theta}(x) = 1.

    Take the gradient of both sides of the normalization condition:

    .. math::

        \nabla_{\theta} \int_x P_{\theta}(x) = \nabla_{\theta} 1 = 0.

    Use the log derivative trick to get:

    .. math::

        0 &= \nabla_{\theta} \int_x P_{\theta}(x) \\
        &= \int_x \nabla_{\theta} P_{\theta}(x) \\
        &= \int_x P_{\theta}(x) \nabla_{\theta} \log P_{\theta}(x) \\
        \therefore 0 &= \underE{x \sim P_{\theta}}{\nabla_{\theta} \log P_{\theta}(x)}.

.. [1] The author of this article is not aware of this lemma being given a standard name anywhere in the literature. But given how often it comes up, it seems pretty worthwhile to give it some kind of name for ease of reference.

Don't Let the Past Distract You
===============================

Examine our most recent expression for the policy gradient:

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)}.

Taking a step with this gradient pushes up the log-probabilities of each action in proportion to :math:`R(\tau)`, the sum of *all rewards ever obtained*. But this doesn't make much sense. 

Agents should really only reinforce actions on the basis of their *consequences*. Rewards obtained before taking an action have no bearing on how good that action was: only rewards that come *after*.

It turns out that this intuition shows up in the math, and we can show that the policy gradient can also be expressed by

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})}.

In this form, actions are only reinforced based on rewards obtained after they are taken. 

We'll call this form the "reward-to-go policy gradient," because the sum of rewards after a point in a trajectory,

.. math::

    \hat{R}_t \doteq \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}),

is called the **reward-to-go** from that point, and this policy gradient expression depends on the reward-to-go from state-action pairs.

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

