# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Simple compiler gym tabular q learning example.

Usage: python tabular_q.py --benchmark=<benchmark>

Using selected features from Autophase observation space, given a specific training
program as gym environment, find the best action sequence using online q learning.
"""

import random
from typing import Dict, NamedTuple

import gym
from absl import app, flags

import compiler_gym.util.flags.episode_length  # noqa Flag definition.
import compiler_gym.util.flags.episodes  # noqa Flag definition.
import compiler_gym.util.flags.learning_rate  # noqa Flag definition.
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.timer import Timer

flags.DEFINE_list(
    "tabular_q_actions",
    [
        "-add-discriminators",
        "-adce",
        "-aggressive-instcombine",
        "-alignment-from-assumptions",
        "-always-inline",
        "-argpromotion",
        "-attributor",
        "-barrier",
        "-bdce",
        "-break-crit-edges",
        "-simplifycfg",
        "-callsite-splitting",
        "-called-value-propagation",
        "-canonicalize-aliases",
        "-consthoist",
        "-constmerge",
        "-constprop",
        "-coro-cleanup",
        "-coro-early",
        "-coro-elide",
        "-coro-split",
        "-correlated-propagation",
        "-cross-dso-cfi",
        "-deadargelim",
        "-dce",
        "-die",
        "-dse",
        "-reg2mem",
        "-div-rem-pairs",
        "-early-cse-memssa",
        "-elim-avail-extern",
        "-ee-instrument",
        "-flattencfg",
        "-float2int",
        "-forceattrs",
        "-inline",
        "-insert-gcov-profiling",
        "-gvn-hoist",
        "-gvn",
        "-instcombine",
        "-globaldce",
        "-globalopt",
        "-globalsplit",
        "-guard-widening",
        "-hotcoldsplit",
        "-ipconstprop",
        "-ipsccp",
        "-indvars",
        "-irce",
        "-infer-address-spaces",
        "-inferattrs",
        "-inject-tli-mappings",
        "-instsimplify",
        "-instcombine",
        "-instnamer",
        "-jump-threading",
        "-lcssa",
        "-licm",
        "-libcalls-shrinkwrap",
        "-load-store-vectorizer",
        "-loop-data-prefetch",
        "-loop-deletion",
        "-loop-distribute",
        "-loop-fusion",
        "-loop-guard-widening",
        "-loop-idiom",
        "-loop-instsimplify",
        "-loop-interchange",
        "-loop-load-elim",
        "-loop-predication",
        "-loop-reroll",
        "-loop-rotate",
        "-loop-simplifycfg",
        "-loop-simplify",
        "-loop-sink",
        "-loop-reduce",
        "-loop-rotate",
        "-loop-unroll-and-jam",
        "-loop-unroll",
        "-loop-unswitch",
        "-loop-vectorize",
        "-loop-versioning-licm",
        "-loop-versioning",
        "-loweratomic",
        "-lower-constant-intrinsics",
        "-lower-expect",
        "-lower-guard-intrinsic",
        "-lowerinvoke",
        "-lower-matrix-intrinsics",
        "-lowerswitch",
        "-lower-widenable-condition",
        "-memcpyopt",
        "-mergefunc",
        "-mergeicmps",
        "-mldst-motion",
        "-sancov",
        "-name-anon-globals",
        "-nary-reassociate",
        "-newgvn",
        "-pgo-memop-opt",
        "-partial-inliner",
        "-partially-inline-libcalls",
        "-post-inline-ee-instrument",
        "-functionattrs",
        "-mem2reg",
        "-newgvn",
        "-prune-eh",
        "-reassociate",
        "-redundant-dbg-inst-elim",
        "-reg2mem",
        "-rpo-functionattrs",
        "-rewrite-statepoints-for-gc",
        "-sccp",
        "-simplifycfg",
        "-slp-vectorizer",
        "-sroa",
        "-scalarizer",
        "-separate-const-offset-from-gep",
        "-simple-loop-unswitch",
        "-sink",
        "-speculative-execution",
        "-slsr",
        "-strip-dead-prototypes",
        "-strip-debug-declare",
        "-strip-nondebug",
        "-strip",
        "-tailcallelim",
        "-mergereturn",
    ],
    "A list of action names to explore from.",
)
flags.DEFINE_float("discount", 1.0, "The discount factor.")
flags.DEFINE_list(
    "features_indices",
    [19, 22, 51],
    "Indices of Alphaphase features that are used to construct a state",
)
flags.DEFINE_integer(
    "log_every", 50, "number of episode interval where progress is reported."
)
flags.DEFINE_float("epsilon", 0.2, "Epsilon rate of exploration. ")
FLAGS = flags.FLAGS


class StateActionTuple(NamedTuple):
    """An state action tuple used as q-table keys"""

    autophase0: int
    autophase1: int
    autophase2: int
    cur_step: int
    action_index: int


def make_q_table_key(autophase_feature, action, step):
    """Create a hashable Q-table key.

    For tabular learning we will be constructing a Q-table which maps a
    (state, action) pair to an expected (remaining) reward. The purpose of this
    function is to convert the (state, action) properties into a hashable tuple
    that can be used as a key for a Q-table dictionary.

    In the CompilerGym setup, encoding the true state the program is not obvious,
    and this solution turns to use the observations from Autophase features instead.
    The default arguments handpicked 3 indices from the Autophase feature that
    appear to change a lot during optimization.

    In addition, the current step in the episode is added to the state representation
    as well. In the current fixed-episode-length setup, we need to differentiate
    reaching a state at different steps, as they can lead to different final rewards,
    depending on the remaining optimization steps.

    Finally, we add the action index to the key.
    """
    return StateActionTuple(
        *autophase_feature[FLAGS.features_indices],
        step,
        FLAGS.tabular_q_actions.index(action),
    )


def select_action(q_table, ob, step, epsilon=0.0):
    qs = [
        q_table.get(make_q_table_key(ob, act, step), -1)
        for act in FLAGS.tabular_q_actions
    ]
    if random.random() < epsilon:
        return random.choice(FLAGS.tabular_q_actions)
    max_indices = [i for i, x in enumerate(qs) if x == max(qs)]
    # Breaking ties at random by selecting any of the indices.
    return FLAGS.tabular_q_actions[random.choice(max_indices)]


def get_max_q_value(q_table, ob, step):
    max_q = 0
    for act in FLAGS.tabular_q_actions:
        hashed = make_q_table_key(ob, act, step)
        max_q = max(q_table.get(hashed, 0), max_q)
    return max_q


def rollout(qtable, env, printout=False):
    # rollout the policy using a given Q table greedily.
    observation = env.reset()
    action_seq, rewards = [], []
    for i in range(FLAGS.episode_length):
        a = select_action(qtable, observation, i)
        action_seq.append(a)
        observation, reward, done, info = env.step(env.action_space.flags.index(a))
        rewards.append(reward)
        if done:
            break
    if printout:
        print(
            "Resulting sequence: ", ",".join(action_seq), f"total reward {sum(rewards)}"
        )
    return sum(rewards)


def train(q_table, env):
    # Buffer an old version of q table to inspect training progress.
    prev_q = {}

    # Run the training process "online", where the policy evaluation and
    # policy improvement happens directly after one another.
    for i in range(1, FLAGS.episodes + 1):
        current_length = 0
        observation = env.reset()
        while current_length < FLAGS.episode_length:
            # Run epsilon greedy policy to allow exploration.
            a = select_action(q_table, observation, current_length, FLAGS.epsilon)
            hashed = make_q_table_key(observation, a, current_length)
            if hashed not in q_table:
                q_table[hashed] = 0
            # Take a stap in the environment, record the reward and state transition.
            # Effectively we are evaluating the policy by taking a step in the
            # environment.
            observation, reward, done, info = env.step(env.action_space.flags.index(a))
            if done:
                break
            current_length += 1

            # Compute the target value of the current state, by using the current
            # step-reward and bootstrapping from the next state. In Q-learning,
            # a greedy policy is implied by the Q-table, thus we can approximate
            # the expected reward at the next state as the maximum value of
            # all the associated state-action pair rewards (Q values). A discount
            # can be used to emphasize on immediate early rewards, and encourage
            # the agent to achieve higher rewards sooner than later.
            target = reward + FLAGS.discount * get_max_q_value(
                q_table, observation, current_length
            )

            # Update Q value. Instead of replacing the Q value at the current
            # state action pair directly, a learning rate is introduced to interpolate
            # between the current value and target value, effectively damping the
            # changes. By updating the Q-table, we effectively updated the policy.
            q_table[hashed] = (
                FLAGS.learning_rate * target
                + (1 - FLAGS.learning_rate) * q_table[hashed]
            )

        if FLAGS.log_every and i % FLAGS.log_every == 0:

            def compare_qs(q_old, q_new):
                diff = [q_new[k] - v for k, v in q_old.items()]
                return sum(diff) / len(diff) if diff else 0.0

            difference = compare_qs(prev_q, q_table)
            # Evaluate the current policy
            cur_rewards = rollout(q_table, env)
            print(
                f"episode={i:4d}, cur_reward={cur_rewards:.5f}, Q-table_entries={len(q_table):5d}, Q-table_diff={difference:.7f}"
            )
            prev_q = q_table.copy()


def main(argv):
    # Initialize a Q table.
    q_table: Dict[StateActionTuple, float] = {}
    benchmark = benchmark_from_flags()
    assert benchmark, "You must specify a benchmark using the --benchmark flag"

    with gym.make("llvm-ic-v0", benchmark=benchmark) as env:
        env.observation_space = "Autophase"

        # Train a Q-table.
        with Timer("Constructing Q-table"):
            train(q_table, env)

        # Rollout resulting policy.
        rollout(q_table, env, printout=True)


if __name__ == "__main__":
    app.run(main)
