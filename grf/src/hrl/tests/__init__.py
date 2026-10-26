"""One test file per node: test_nNN_<unit>.py. Read them: they are part of what you learn from.

Each file has
  NODE, TITLE, LAYER, NEEDS    where it sits in the build map
  test_unit_*()                this node's own code, ALONE: anything from a lower node is
                               swapped for a fake with fixed answers (see _support.swap), so a
                               unit test can only fail because of this node's code
  test_stack()                 this node's code on top of YOUR real code for everything below
                               it. Units pass but stack fails -> the bug is below, or in the join
  demo()                       optional: prints something to predict before you run it
Run through the runner, from grf/src:  python3 -m hrl.check 24
"""
