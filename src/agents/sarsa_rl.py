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
    "optimization_passes",
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
    autophase0: int
    autophase1: int
    autophase2: int
    cur_step: int
    action_index: int


def make_key_tuple(autophase_feature, action, step):
    return StateActionTuple(
        *autophase_feature[FLAGS.features_indices],
        step,
        FLAGS.optimization_passes.index(action),
    )


def select_action(state_dict, ob, step, epsilon=0.0):
    qs = [
        state_dict.get(make_key_tuple(ob, act, step), -1)
        for act in FLAGS.optimization_passes
    ]
    if random.random() < epsilon:
        return random.choice(FLAGS.optimization_passes)
    max_indices = [i for i, x in enumerate(qs) if x == max(qs)]
    return FLAGS.optimization_passes[random.choice(max_indices)]


def get_max_q_value(state_dict, ob, step):
    max_q = 0
    for act in FLAGS.optimization_passes:
        hashed = make_key_tuple(ob, act, step)
        max_q = max(state_dict.get(hashed, 0), max_q)
    return max_q


def rollout(state_dict, env, printout=False):
    observation = env.reset()
    action_seq, rewards = [], []
    for i in range(FLAGS.episode_length):
        a = select_action(state_dict, observation, i)
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


def train(state_dict, env):
    prev_q = {}

    for i in range(1, FLAGS.episodes + 1):
        current_length = 0
        observation = env.reset()

        a = select_action(state_dict, observation, current_length, FLAGS.epsilon)
        
        while current_length < FLAGS.episode_length:
            hashed = make_key_tuple(observation, a, current_length)
            if hashed not in state_dict:
                state_dict[hashed] = 0
            
            next_observation, reward, done, info = env.step(env.action_space.flags.index(a))
            current_length += 1

            if not done and current_length < FLAGS.episode_length:
                a_next = select_action(state_dict, next_observation, current_length, FLAGS.epsilon)
                hashed_next = make_key_tuple(next_observation, a_next, current_length)
                if hashed_next not in state_dict:
                    state_dict[hashed_next] = 0
                q_next = state_dict[hashed_next]
            else:
                q_next = 0

            target = reward + FLAGS.discount * q_next

            state_dict[hashed] = (
                FLAGS.learning_rate * target
                + (1 - FLAGS.learning_rate) * state_dict[hashed]
            )

            if done:
                break

            observation = next_observation
            a = a_next

        if FLAGS.log_every and i % FLAGS.log_every == 0:
            def compare_qs(q_old, q_new):
                diff = [q_new[k] - v for k, v in q_old.items()]
                return sum(diff) / len(diff) if diff else 0.0

            difference = compare_qs(prev_q, state_dict)
            cur_rewards = rollout(state_dict, env)
            print(
                f"episode={i:4d}, cur_reward={cur_rewards:.5f}, entries={len(state_dict):5d}, diff={difference:.7f}"
            )
            prev_q = state_dict.copy()



def main(argv):
    states: Dict[StateActionTuple, float] = {}
    benchmark = benchmark_from_flags()
    assert benchmark, "You must specify a benchmark using the --benchmark flag"

    with gym.make("llvm-ic-v0", benchmark=benchmark) as env:
        env.observation_space = "Autophase"

        with Timer("Running training..."):
            train(states, env)

        rollout(states, env, printout=True)


if __name__ == "__main__":
    app.run(main)
