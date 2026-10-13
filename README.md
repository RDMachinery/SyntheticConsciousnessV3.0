# Weightless Consciousness: Aleksander's Five Axioms as a WISARD-Style Neural State Machine

A single-file Java program that implements Igor Aleksander's five axioms of a synthetic mind using **weightless neural networks**: RAM neurons that are addressed by bits and trained by writing into memory, in the tradition of Aleksander's WISARD and his neural state machines.

It is the sequel to the tabular version in this repository (`SyntheticConsciousness.java`). The task, the ablation experiment and the switches are the same, but the lookup tables have been replaced by networks, so the two can be compared directly.

There are no dependencies.


## Why this exists

Aleksander's systems were not hash maps. They were networks of small RAM look-up tables, with state held as a pattern across the network and learning done by writing to memory. This program moves the axioms onto that kind of substrate and then tests, by switching each axiom off in turn, whether each one still does real work.

The answer turns out to be more interesting than "yes". 

## What this is, and what it isn't

This implements the *functional architecture* described by the axioms. It makes **no claim that the agent has experience**. Satisfying the axioms and being conscious are different questions, and the gap between them is the hard problem.

It is also not a faithful reproduction of Aleksander's hardware:

- His neurons were probabilistic logic nodes that stored 0, 1 or "undefined". These RAMs store counts and running averages, which is a close cousin but not the same thing.
- The affect networks average over their RAMs (a step closer to a CMAC than to a WISARD discriminator).
- The RAM wiring is designed by hand (local windows), not random as in classic WISARD.

## Quick start

You need Java 11 or later. It was developed and tested on OpenJDK 21.

```bash
# Run the ablation experiment (about 15 seconds on one core)
java WeightlessConsciousness.java

# Watch one trained episode, step by step
java WeightlessConsciousness.java trace

# Watch the same episode with an axiom switched off
java WeightlessConsciousness.java trace --no-attention
java WeightlessConsciousness.java trace --no-imagination
java WeightlessConsciousness.java trace --no-planning
java WeightlessConsciousness.java trace --no-emotion
```

Flags can be combined, for example `trace --no-attention --no-emotion`.

To compile first:

```bash
javac WeightlessConsciousness.java
java WeightlessConsciousness
```

## The task

The agent lives in a corridor 15 cells long and has four actions: `LEFT`, `RIGHT`, `EAT` and `REST`.

- **Channel 0** is a food gradient telling the agent how far away the food is and in which direction.
- **Channels 1 to 3** are pure noise, with new random values every step.
- Moving costs a little (-0.02), eating in the wrong place is penalised (-0.1), resting is free, and eating at the food earns +1. The food then respawns elsewhere.
- Each episode is 40 steps, with a random start and a random food location.

The agent is never told which channel matters, what its actions do, or where reward comes from.

## What "weightless" means here

In a conventional neural network, neurons compute weighted sums of their inputs, and learning adjusts the weights. In a weightless network, each neuron is a small **RAM** (a look-up table):

1. A few input bits form an **address**.
2. **Learning** writes something into the location at that address.
3. **Recall** reads the location back.

There is no arithmetic between neurons. Generalisation comes from overlap: two similar inputs differ in some bits but not others, so they share addresses in some of the RAMs and give partly shared answers.

## How each axiom is built

Aleksander numbers the axioms 1 Presence, 2 Imagination, 3 Attention, 4 Planning, 5 Emotion. The code keeps his numbers in its labels but runs them in the order the data flows, since attention decides what the later stages see.

| Axiom | Implementation | Size |
|---|---|---|
| **1 Presence** | The world is encoded as a *retina* of bits. Each channel uses a thermometer code: a level of *k* switches on bits 0 to *k*-1, so nearby values give similar patterns. | 4 channels x 28 bits |
| **3 Attention** | Tracks how much each channel changes from step to step and attends to the smooth ones. An unattended channel stays blank on the retina and its RAM neurons stop taking part. | 1 mask per step |
| **2 Imagination** | A **neural state machine**: one RAM neuron per retina bit, addressed by the action (4 bits, one-hot) and the bit's own neighbourhood (3 bits). Each stores how often its bit was next on and off. Feeding its predictions back in as its own input is the free-running imagination of Aleksander's architecture. | 112 RAM neurons, 128 addresses each |
| **4 Planning** | Searches three imagined moves ahead through the neural state machine and picks the first move of the best sequence. | Depth 3 |
| **5 Emotion** | Two RAM nets that read 8-bit windows of the retina: one learns the **value** of a state, the other what each action tends to bring. A **mood** (a running sense of how well things are going) raises the exploration rate when things go badly. | 6 windows per channel |

A reading note for `trace` mode: a retina bar of length 14 means the food is exactly here. Longer means the food is to the right, shorter means it is to the left.

## The ablation experiment

Running `java WeightlessConsciousness.java` trains 20 agents for 300 episodes each, for every configuration, and reports reward per episode. The results below come from that run (mean over 20 runs, with standard error). Random number generators are seeded, so results are reproducible.

| Configuration | First 50 episodes | Last 50 episodes | RAM locations written |
|---|---|---|---|
| All five axioms | +1.90 +/- 0.20 | +4.85 +/- 0.04 | 1785 |
| No attention | +2.36 +/- 0.15 | +4.79 +/- 0.04 | 2839 |
| No imagination | -0.09 +/- 0.02 | -0.01 +/- 0.02 | 614 |
| No planning | +2.15 +/- 0.20 | +4.82 +/- 0.05 | 1787 |
| No emotion | +0.53 +/- 0.04 | +0.42 +/- 0.02 | 1611 |
| Scripted walk-to-food (ceiling) | +5.43 | | |

### Comparison with the tabular version

| Configuration | Tabular, last 50 | Weightless, last 50 |
|---|---|---|
| All five axioms | +4.76 | +4.85 |
| No attention | -1.05 | +4.79 |
| No imagination | -0.01 | -0.01 |
| No planning | +4.80 | +4.82 |
| No emotion | +0.33 | +0.42 |

The weightless agent learns more slowly at first (+1.90 versus +3.01 over the first 50 episodes) but reaches the same level.

## What the results say

- **Imagination and emotion are essential.** Without imagination the agent is purely reactive and concludes that doing nothing is best. Without emotion it has no sense of which imagined states are good and can only see as far as remembered immediate rewards.
- **Attention no longer matters for reward.** The tabular agent collapsed without attention, because every noisy state looked new. The weightless network tolerates the noise because each neuron only looks at a small window of one channel. Here attention saves memory and computation (1785 locations versus 2839) but not performance. This is a finding about the architecture: generalisation absorbs a lot of distraction.
- **Planning is still not essential.** It makes no reliable difference here, because the corridor is simple enough for single-step lookahead to do well. A world with longer horizons or obstacles is needed to test it properly.

### An exploratory test: more distraction

A quick additional run with **15 noise channels** instead of 3 (8 runs of 300 episodes) suggests attention does become load-bearing as the distraction grows:

| Configuration | First 50 episodes | Last 50 episodes | RAM locations written |
|---|---|---|---|
| All five axioms | +2.09 +/- 0.22 | +5.01 +/- 0.04 | 6007 |
| No attention | +0.44 +/- 0.17 | +4.57 +/- 0.12 | 11360 |

This is exploratory, with a small number of runs and no dedicated switch in the program. To reproduce it, change `CHANNELS` to `16` and call `ablationExperiment(8, 300)` in `main`. Channels 1 and up are filled with noise automatically.

## Watching the agent think

`trace` mode shows the five axioms at work on each step of a trained agent:

```
step 1   (food is +8 cells away)
  [1 Presence]    sensors = [+0.57 +0.34 -0.27 -0.71]
  [3 Attention]   attending to channels [0]
                  retina ch0 = ######################......
  [2 Imagination] predicted ch0 after each action:
                  LEFT -> #######################.....
                  RIGHT -> #####################.......
                  EAT -> ######################......
                  REST -> ######################......
  [4 Planning]    lookahead 3:  LEFT +0.19   RIGHT +0.29   EAT +0.17   REST +0.27
  [5 Emotion]     mood +0.83   value of this state +0.28   explore-rate 0.07
  => RIGHT
```

The agent ignores the three noise channels, uses its neural state machine to predict the retina after each action (moving right shortens the bar, bringing it closer to 14), scores the imagined futures, and heads for the food. Compare with `trace --no-attention`, where all four channels are attended, and `trace --no-imagination`, where the prediction line is blank.

## Parameters

These constants sit at the top of `WeightlessConsciousness.java`.

| Constant | Default | Meaning |
|---|---|---|
| `GAMMA` | 0.9 | How much future reward counts relative to immediate reward |
| `RATE` | 0.4 | How far a RAM moves towards what it just saw |
| `OPTIMISM` | 0.05 | Unwritten memory looks mildly promising, which drives exploration |
| `EPISODE_STEPS` | 40 | Length of each episode |
| `PLAN_DEPTH` | 3 | How many imagined moves the planner looks ahead |
| `CHANNELS` | 4 | Sensor channels (one signal, the rest noise) |
| `BITS` | 28 | Retina bits per channel |

The attention threshold (`STABLE_BELOW`) and warm-up length (`WARMUP`) are inside the `Agent` class, and the window size and stride of the affect networks are inside `AffectNet`.

## Limitations

- **Not strictly weightless in Aleksander's sense.** The state machine stores counts and the affect nets store real values and average them, instead of using 0/1/undefined nodes.
- **Hand-designed wiring.** The RAMs read local windows of a channel's retina. Classic WISARD wires its RAMs to random bits.
- **The retina is matched to the world.** The thermometer resolution is chosen so that one corridor cell is one retina level.
- **The attention rule is a heuristic.** It prefers channels that change smoothly. A smooth but irrelevant channel would fool it, which is the subject of the next planned experiment.
- **The emotion switch bundles two things:** state values and mood-driven exploration.
- **The axioms interlock.** Imagination is useless without some way to value its output, so the ablations are not clean single-factor tests.
- **The world is small and fully learnable.**

## Ideas for extending it

- Add a **smooth decoy channel** to test whether the attention heuristic can be fooled, then replace the heuristic with attention that learns relevance from reward.
- Add a harder environment (obstacles, a moving predator) so that planning matters at the end of training.
- Split the emotion switch into separate "affective value" and "mood" switches.
- Replace the averaging read-out with true WISARD discriminators, one per class, and compare.
- Replace the count-based neurons with probabilistic logic nodes (0, 1, undefined) as in Aleksander's later work.

## Background

- Aleksander, I., Thomas, W. V. and Bowden, P. A. (1984). *WISARD: a radical step forward in image recognition.* Sensor Review, 4(3), 120-124.
- Aleksander, I. and Dunmall, B. (2003). *Axioms and Tests for the Presence of Minimal Consciousness in Agents.* Journal of Consciousness Studies, 10(4-5), 7-18.
- Aleksander, I. (1996). *Impossible Minds: My Neurons, My Consciousness.* Imperial College Press.
- Aleksander, I. (2005). *The World in My Mind, My Mind in the World: Key Mechanisms of Consciousness in People, Animals and Machines.* Imprint Academic.

## License

Add your preferred license here (MIT is a common choice for small projects like this).
