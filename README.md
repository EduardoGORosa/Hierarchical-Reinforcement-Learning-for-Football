# Hierarchical Reinforcement Learning for Football

Undergraduate thesis (TCC) in progress at INF-UFRGS, advised by Prof. Dr. Anderson Rocha Tavares.

## Theme

**Decomposição Hierárquica e Eficiência Amostral em Aprendizado por Reforço para Futebol Simulado**

> Agentes treinados por aprendizado por reforço de ponta a ponta tendem a tratar movimentação e decisão tática como um único problema de otimização, o que amplia o espaço de busca e eleva o custo computacional do treinamento. Este trabalho investiga se a decomposição hierárquica dessas decisões — em um nível de movimentação e um nível de decisão tática — melhora a eficiência amostral e o desempenho de agentes de futebol simulado, em comparação a agentes treinados de forma não hierárquica. O nível de movimentação será obtido por aprendizado por imitação a partir de dados reais de rastreamento de jogadores, e o nível tático por aprendizado por reforço, avaliados no ambiente Google Research Football.

In English: end-to-end trained agents tend to treat movement and tactical decisions as a single optimization problem, which widens the search space and raises training cost. This thesis tests whether decomposing those decisions into a movement layer and a tactics layer improves sample efficiency and performance compared to a non-hierarchical agent. The movement layer is trained via imitation learning from real player tracking data; the tactics layer via reinforcement learning; both evaluated in Google Research Football.
