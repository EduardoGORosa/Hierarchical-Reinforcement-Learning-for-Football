"""Hierarchical agent for GRF, built on the options framework as the HRL survey sets it out.

Framework: Klissarov, Bagaria, Luo, Konidaris, Precup & Machado, "Discovering Temporal
Structure: An Overview of Hierarchical Reinforcement Learning" (arXiv:2506.14045). The code
follows its Algorithm 1: OptionSpecification (options.py) -> OptionLearning and Interact
(execution.py) -> PolicyImprovement of mu (train_hrl.py, smdp.py). Style steering (style.py)
acts on mu, and evaluate.py is the protocol.

Built as a ladder of nodes (Obsidian: sessions/hierarchical-rl-football/build-map.md). Each
TODO(node N) is one unit. Its tests are in tests/test_nNN_*.py: UNIT tests run it alone, with
lower nodes faked; the STACK test runs it on top of your real code for everything below.

    cd grf/src && python3 -m hrl.check           # the ladder (host: layers 1-3, 6-7)
    ./run.sh hrl check                           # the same in the container: every layer

    layer  module         nodes            what you write
    1      pg_basics.py   23 24 22 25 26   log-gradient by nudge, one episode's contribution,
                                           reweighting old data, PPO clip, a baseline
    2      smdp.py        27 28 29         geometric sum, one transition of mu, GAE with gamma^tau
    3      options.py     30 31            subgoals and option reward r_o (Def. 3); beta, with the
                                           non-Markov clock or the Markov geometric form
    4      execution.py   32 33 34 35 36   call-and-return; reward mixing (8.3); option-learning
                                           env with z as input; interruption; options + primitives
    5      train_hrl.py   37               two phases (8.1): provided, read and changed, not written
    6      style.py       38 39 40 43 44   softmax, the style rule, KL, what PPO must store
                                           (41, 42 are theory nodes)
    7      evaluate.py    45 46 47         distinct game seeds, IQM and bootstrap, potential shaping

    toy.py   KnownAnswerPitch: a stand-in for GRF with exact answers, used by layers 4-7
    check.py the runner          tests/_support.py  swap() and the other test helpers
"""
