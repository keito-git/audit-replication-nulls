# Audit replication and distribution-matched nulls

Code and stored runs for *Verification and Validation of Social Simulation with
Large Language Models: Audit Replication and Distribution-Matched Nulls*.

The procedure validates an LLM social simulation by whether it reproduces the
directional findings of a published algorithm audit, and then measures, with
baselines that hold only the observed agent-side behaviour, how much of a
successful reproduction the agents alone already explain.

## Layout

```
code/simulation/   the simulation, the recommender, the pre-expenditure tests
                   and the verdict
code/tables/       every table and figure in the paper, generated from the
                   stored runs
data/              the frozen content pool and all stored runs
output/            where the generators write
```

## Reproducing the tables and figures

No credentials and no model calls are needed: the generators read the stored
runs.

```
cd code/tables
python3 make_tables.py            # verdict, adherence, split replication
python3 null_agent.py             # the pre-expenditure test of the environment
python3 stability.py              # repeated splits, equivalence, Figure 4
python3 sensitivity.py            # aggregation windows
python3 environment_compare.py    # tally recommender against a contextual bandit
python3 prompt_styles.py          # explicit, persona, behaviour history
python3 second_control.py         # the two meaningless controls
python3 distribution_check.py     # a distribution-level criterion on the same runs
python3 mediation.py              # amplification regressed on agent-side quantities
python3 application.py            # the regime with no audit finding
python3 make_figures.py           # Figures 2 and 3
python3 make_fig1.py              # Figure 1
python3 attribution_ladder.py --from-cache
```

`attribution_ladder.py` without `--from-cache` recomputes the ladder, which
takes hours across four models at one hundred seeds. `--only NAME` restricts it
to one model, which is enough to check that a change still reproduces the
stored numbers.

## Re-running the simulation

This calls a generative model and costs money. Set a key first:

```
export OPENROUTER_API_KEY=...
cd code/simulation
AUDITREP_SEEDS=100 AUDITREP_MODEL=mistralai/mistral-small-24b-instruct-2501 \
  python3 gate.py run pref_neg pref_pos pref_h
python3 gate_verdict.py
```

Environment variables: `AUDITREP_SEEDS` the number of seeds, `AUDITREP_MODEL`
the model identifier, `AUDITREP_PROMPT` one of `explicit`, `persona`,
`behaviour`.

The content pool in `data/pool.json` is frozen. `gate.py` refuses to regenerate
it, so every condition, seed and model sees the same items.

## What the pre-expenditure tests do

`platform_test.py` and `code/tables/null_agent.py` replace the generative model
by a chooser that follows the assigned preference with a fixed probability,
identical in every condition. If the feature space of the environment is
symmetric the conditions must agree; a spread is a property of the
representation and not of the agents. `calib_null.py` measures whether the
signal to be learned is separable from its noise at the available number of
observations. Neither makes a model call, so both can be run before any
expenditure.

## Verdict

The verdict takes four values. Void when the treatment was not delivered,
inconclusive when the direction is stable but the power is insufficient,
failure when the direction is not stable, and pass otherwise.
