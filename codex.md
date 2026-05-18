# 多智能体基础个人项目

> This Docs is for codex to refer to, which detaily describe the target of my project

格子游戏上的单智能体强化学习
概述
在这个项目中，你将探索基于价值的表格方法和深度策略梯度方法之间的根本区别。通过从头
开始实现算法并在复杂性不断增加的环境中进行测试，你将亲身体验“维度灾难”，并了解深度强化
学习（DRL）如何克服离散状态表示的局限性。
环境：MiniGrid 动态障碍物
你将使用MiniGrid-Dynamic-Obstacles 系列环境（https://minigrid.farama.org/environments/
minigrid/DynamicObstaclesEnv/）。在这个网格世界中，智能体必须在避开移动障碍物的同时导
航到目标方块。请将在四个难度递增的变体中测试你的算法：
1. MiniGrid-Dynamic-Obstacles-5x5-v0
2. MiniGrid-Dynamic-Obstacles-Random-5x5-v0
3. MiniGrid-Dynamic-Obstacles-8x8-v0
4. MiniGrid-Dynamic-Obstacles-16x16-v0
包装器：默认情况下，MiniGrid-Dynamic-Obstacles 返回RGB 形式的局部观测。对于本项目，
请使用SymbolicObsWrapper（https://minigrid.farama.org/api/wrappers/#symbolic-obs）来
包装环境。此包装器将原始网格数据转换为符号化表示，然后你可以使用该观测，或者你可以考虑
进一步转化你的观测成更简单的形式。
## 任务1：MDP 形式化与环境分析
在编写任何训练循环之前，你必须将环境形式化为马尔可夫决策过程(MDP)。你下面的分析应
当基于经过SymbolicObsWrapper 处理后的环境。
在最终报告中，提供以下组件的详细明细：
• 状态空间(S)：符号包装器如何表示状态？计算5x5、8x8 和16x16 环境的理论状态空间大小。
解释移动障碍物的存在如何影响可能状态的总数。
• 动作空间(A)：列出智能体可用的离散动作。
• 奖励函数(R)：描述环境的奖励结构。它是密集的还是稀疏的？
1
• 转移概率(P)：转移是确定性的还是随机的？动态障碍物如何表现？
## 任务2：算法实现
你必须从头开始实现两种强化学习算法。严格禁止使用外部强化学习库（例如Stable-Baselines3、
Ray RLlib、CleanRL）。你只能使用标准库（如NumPy）和深度学习框架（PyTorch）来构建神经
网络。
1. 表格型Q-Learning：实现带有ϵ-贪婪探索策略的标准Q 表。
2. 近端策略优化(PPO)：实现PPO-Clip 算法。你的实现应包括Actor-Critic 神经网络架构、广
义优势估计(GAE) 以及裁剪的替代目标函数（surrogate objective function）。
重要设计约束（最大步数）：强化学习算法有时难以收敛，特别是大型网格上的表格型Q-Learning。
为了防止无限循环并管理计算时间，你必须设计训练循环，使其在达到最大总训练步数时自动停止。
如果算法在此限制内未能收敛，请记录截至该点的性能并评估其失败的原因。
## 任务3：实验评估与分析
在四个指定的环境变体上训练你的两个智能体。确保你针对每种环境规模适当地调整了超参数
（学习率、ϵ-衰减、折扣因子、PPO 裁剪范围等）。
在你的报告中，提供解决以下问题#的比较分析：
1. 学习曲线：绘制两种算法在所有四个环境中的回合回报与时间步数的关系图。你的结果应至少
是3 个随机种子的平均值。
2. 维度灾难：讨论当环境从静态5x5 扩展到16x16 时，表格型Q-Learning 的表现如何。训练时
间和成功率如何变化？
3. 泛化能力：与表格型Q-Learning 相比，PPO 如何处理Random-5x5-v0 中随机化的起始位置？
4. 计算效率：比较两种算法在不同难度级别下的样本效率（达到最优策略所需的步数）。
提交内容
1. 源代码：结构清晰的代码库或ZIP 文件，包含你的环境设置、算法实现和训练脚本。代码必须
有良好的注释。
2. 项目报告：一份PDF 文档，包含你的MDP 分析、超参数详细信息、绘制的学习曲线以及来
自任务3 的分析讨论。
2