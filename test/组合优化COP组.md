# 组合优化COP

## 什么是COP?（仅供参考）

或许你听过大名鼎鼎的AlphaGP，其将深度神经网络与树搜索算法(Monte Carlo Tree Search)进行了有效的结合，来优化和搜索下棋的最佳策略。AlphaGo便可以视为一个组合优化的成功例子。

关于组合优化，可以参考以下内容：

### Wiki

[Combinatorial optimization](https://en.wikipedia.org/wiki/Combinatorial_optimization)

### ChatGPT

组合优化是一种解决离散优化问题的方法，它涉及在给定的搜索空间中寻找最优的组合，以满足特定的约束条件和优化目标。这里的“组合”是指从一组可选项中选择一些元素的过程。

在组合优化中，通常存在一个目标函数，它描述了待优化的问题，并且需要最小化或最大化这个目标函数。同时，问题可能会有一些限制条件，例如资源约束、边界约束或逻辑约束，这些约束条件必须满足。

举例来说，如果我们有一个机器学习模型，并且希望找到最优的超参数配置，以达到最佳的性能，那么这个超参数优化问题可以被看作是一个组合优化问题。我们需要在超参数空间中搜索，找到一组最佳的超参数值，使得模型的性能最优。

组合优化问题通常是复杂的，因为随着搜索空间的增加，可能会有非常多的组合可能性，使得问题变得非常庞大。许多组合优化问题在一般情况下是NP-hard问题，意味着在多项式时间内找到全局最优解可能是不可行的。因此，研究者通常会使用启发式算法、近似算法或元启发式算法来找到较优解。

在机器学习中，组合优化有许多应用，例如特征选择、超参数优化、集成学习等。通过有效地解决组合优化问题，我们可以提高机器学习模型的性能、效率和泛化能力。

### CMU

[Combinatorial Optimization ](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/glossary/comb.html)

## 学习资源（仅供参考）

### Discrete Optimization|Coursera

[Discrete Optimization](https://zh.coursera.org/learn/discrete-optimization)

### *Combinatorial Optimization: Algorithms and Complexity*

《组合最优化：算法与复杂性》

### Combinatorial Optimization|MIT

[Combinatorial Optimization](https://ocw.mit.edu/courses/18-433-combinatorial-optimization-fall-2003/)

### 部分论文（包括Survey Papers）

- [Reinforcement Learning for Combinatorial Optimization: A Survey](https://arxiv.org/abs/2003.03600)

- [Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon](https://arxiv.org/pdf/1811.06128.pdf)

- [Reinforcement Learning for Combinatorial Optimization: A Survey](https://arxiv.org/pdf/2003.03600.pdf)

- Ptr-Net的基本原理 [Pointer Networks](https://arxiv.org/pdf/1506.03134.pdf)

- [Neural combinatorial optimization with reinforcement learning](https://arxiv.org/pdf/1611.09940.pdf) & [code](https://github.com/pemami4911/neural-combinatorial-rl-pytorch)

- [Reinforcement learning for solving vehicle routing problem](https://arxiv.org/pdf/1802.04240.pdf) & [code](https://github.com/mveres01/pytorch-drl4vrp)

- Attention: Learn to solve routing problems [ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!](https://openreview.net/pdf?id=ByxBFsRqYm)
- [基于深度强化学习的组合优化研究进展 - 中国知网 (cnki.net)](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2022&filename=MOTO202111002&uniplatform=NZKPT&v=F2rgybyahtVidDbK8Nqi6_DL4n9_zZQnjAgzMlkXFijBa-BmglsMbwg9PB6gzFig)

## TSP

我们以TSP为例：

TSP(Traveling Salesman Problem, 旅行商问题)是组合优化中的一个经典NP困难问题。

TSP问题可以描述为:

给定N个城市和每对城市之间的距离，要求找到一条访问每一个城市一次并回到起点的路线，使得整个路线的长度最短。

这是一个典型的组合优化问题,需要从N!种可能的路线中选择出最优解。

在算法课中，你可能使用过O(n^2 2^n)的动态规划求解该问题。

### Solving the Traveling Salesman Problem with Reinforcement Learning

[Solving the Traveling Salesman Problem with Reinforcement Learning](https://ekimetrics.github.io/blog/2021/11/03/tsp/)

以下是一些机器学习解决TSP问题的论文，你应当阅读至少一部分文章，适当地做学习笔记，对模型和算法理解后，可以复现的，选择性地进行复现。我们会就此问题和你进行交流，当然，这十篇论文仅仅是一小部分，你也可以发散性地阅读其他关于TSP的论文。此外，如果你研究了其他组合优化问题，也欢迎届时展示。

### Papers

- **Learning Combinatorial Optimization Algorithms over Graphs.** NeurIPS, 2017 [paper](https://arxiv.org/abs/1704.01665)

- **Learning Heuristics for the TSP by Policy Gradient** CPAIOR, 2018. [paper](https://link.springer.com/chapter/10.1007/978-3-319-93031-2_12), [code](https://github.com/MichelDeudon/encode-attend-navigate)

- **Attention, Learn to Solve Routing Problems!** ICLR, 2019. [paper](https://arxiv.org/abs/1803.08475)

- **Learning to Solve NP-Complete Problems: A Graph Neural Network for Decision TSP.** AAAI, 2019. [paper](https://ojs.aaai.org/index.php/AAAI/article/view/4399)
- **An Efficient Graph Convolutional Network Technique for the Travelling Salesman Problem** Arxiv, 2019. [paper](https://arxiv.org/abs/1906.01227), [code](https://github.com/chaitjo/graph-convnet-tsp)
- **POMO: Policy Optimization with Multiple Optima for Reinforcement Learning.** NeurIPS, 2020. [paper](https://arxiv.org/abs/2010.16011), [code](https://github.com/yd-kwon/POMO/)
- **Learning 2-opt Heuristics for the Traveling Salesman Problem via Deep Reinforcement Learning** ACML, 2020. [paper](http://proceedings.mlr.press/v129/costa20a), [code](https://github.com/paulorocosta/learning-2opt-drl)

- **Learning TSP Requires Rethinking Generalization** CP, 2021. [paper](https://arxiv.org/pdf/2006.07054.pdf), [code](https://github.com/chaitjo/learning-tsp)

- **Graph Neural Network Guided Local Search for the Traveling Salesperson Problem** ICLR, 2022. [paper](https://openreview.net/forum?id=ar92oEosBIg)
- **H-tsp: Hierarchically solving the large-scale traveling salesman problem** AAAI, 2023. [paper](https://www.microsoft.com/en-us/research/publication/h-tsp-hierarchically-solving-the-large-scale-traveling-salesman-problem/), [code](https://github.com/Learning4Optimization-HUST/H-TSP)

关于TSP问题的比赛或开源实践项目较少，不过，你可以随机生成图或二维平面的城市坐标，在本地进行简单的代码实现，来测试你的模型效果。如果你愿意，你还可以将其与动态规划等传统算法的效果进行对比。

与第一轮测试一样，推荐Typora + LaTeX的组合，当然你也可以在个人博客中有所记录，形式不限。
