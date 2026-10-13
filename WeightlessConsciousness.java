import java.util.*;
import java.util.stream.IntStream;

/**
 * Aleksander's five axioms again, this time with the tables replaced by weightless
 * neural networks in the WISARD tradition.
 *
 * "Weightless" means there are no weighted sums between neurons. Each neuron is a
 * small RAM (a look-up table) addressed directly by a handful of input bits. Learning
 * is writing into the RAM; recall is reading from it. Similar inputs share addresses
 * in some of the RAMs, and that sharing is where generalisation comes from.
 *
 * Usage:
 *   java WeightlessConsciousness.java                 run the ablation experiment
 *   java WeightlessConsciousness.java trace           watch one trained episode
 *   java WeightlessConsciousness.java trace --no-attention --no-imagination
 *                                                     (also --no-planning, --no-emotion)
 *
 * The task is the same corridor as before: a food gradient on channel 0, pure noise on
 * channels 1-3, reward for walking to the food and eating.
 *
 * How the axioms are built (Aleksander's numbering in the labels):
 *   1 Presence     the world is encoded as a retina of bits (a thermometer code per channel)
 *   3 Attention    gates which channels reach the retina, and which RAM neurons take part
 *   2 Imagination  a neural state machine: one RAM neuron per retina bit predicts that bit's
 *                  next value from its neighbours and the action. Feeding its output back in
 *                  as its own input is the "free-running" imagination of his architecture.
 *   4 Planning     searches imagined sequences of states
 *   5 Emotion      RAM nets that learn the affective value of states, plus a mood
 *
 * Caveats: Aleksander's neurons were probabilistic logic nodes (stored 0, 1 or "undefined").
 * These RAMs store counts and running averages instead, which is a close cousin but not the
 * same thing. The affect nets read out an average over RAMs, a CMAC-like step that a pure
 * WISARD discriminator would not take.
 */
public class WeightlessConsciousness {

    static final double GAMMA = 0.9;        // how much the future counts
    static final double RATE = 0.4;         // how far a RAM moves towards what it just saw
    static final double OPTIMISM = 0.05;    // unwritten memory looks mildly promising
    static final int EPISODE_STEPS = 40;
    static final int PLAN_DEPTH = 3;

    static final int CHANNELS = 4;
    static final int BITS = 28;             // thermometer bits per channel (levels 0..28)

    enum Action { LEFT, RIGHT, EAT, REST }

    // ------------------------------------------------------------------
    //  The ablation switches
    // ------------------------------------------------------------------
    static final class Config {
        boolean attention = true, imagination = true, planning = true, emotion = true;

        Config without(String axiom) {
            Config c = new Config();
            c.attention = attention; c.imagination = imagination;
            c.planning = planning;   c.emotion = emotion;
            switch (axiom) {
                case "attention"   -> c.attention = false;
                case "imagination" -> c.imagination = false;
                case "planning"    -> c.planning = false;
                case "emotion"     -> c.emotion = false;
                default -> throw new IllegalArgumentException(axiom);
            }
            return c;
        }
    }

    // ------------------------------------------------------------------
    //  The world
    // ------------------------------------------------------------------
    static final class Corridor {
        static final int LENGTH = 15;

        private final Random rng;
        private int pos, food;
        private boolean ate;

        Corridor(Random rng) { this.rng = rng; }

        void reset() {
            pos = rng.nextInt(LENGTH);
            food = rng.nextInt(LENGTH);
            ate = false;
        }

        double[] sense() {
            double[] s = new double[CHANNELS];
            s[0] = (food - pos) / (double) (LENGTH - 1);
            for (int i = 1; i < CHANNELS; i++) s[i] = rng.nextDouble() * 2 - 1;
            return s;
        }

        double step(Action a) {
            ate = false;
            if (a == Action.LEFT)  { pos = Math.max(0, pos - 1);          return -0.02; }
            if (a == Action.RIGHT) { pos = Math.min(LENGTH - 1, pos + 1); return -0.02; }
            if (a == Action.EAT) {
                if (pos == food) {
                    ate = true;
                    do { food = rng.nextInt(LENGTH); } while (food == pos);
                    return 1.0;
                }
                return -0.1;
            }
            return 0.0;                                                    // REST
        }

        boolean ate() { return ate; }
        int distanceToFood() { return food - pos; }
    }

    // ------------------------------------------------------------------
    //  AXIOM 1: PRESENCE. The retina: a thermometer code per channel.
    //  A level of k switches on bits 0..k-1, so nearby values give similar patterns.
    //  A channel is one int; unattended channels stay all-zero.
    // ------------------------------------------------------------------
    static int[] encode(double[] raw, boolean[] attended) {
        int[] retina = new int[CHANNELS];
        for (int c = 0; c < CHANNELS; c++) {
            if (!attended[c]) continue;
            int level = (int) Math.round((raw[c] + 1) / 2.0 * BITS);
            level = Math.max(0, Math.min(BITS, level));
            retina[c] = (1 << level) - 1;
        }
        return retina;
    }

    // ------------------------------------------------------------------
    //  AXIOM 2: IMAGINATION. A neural state machine.
    //  One RAM neuron per retina bit. Its address is seven bits: the action (one-hot,
    //  4 bits) and the bit's own neighbourhood in the retina (3 bits). It remembers how
    //  often the bit was next on and off in that situation.
    // ------------------------------------------------------------------
    static final class NeuralStateMachine {
        private final int[] zeros = new int[CHANNELS * BITS * 128];
        private final int[] ones  = new int[CHANNELS * BITS * 128];

        private static int address(int mask, int bit, int action) {
            int ext = (mask << 1) | 1;               // a constant 1 stands in for "bit -1"
            int neighbourhood = (ext >> bit) & 7;    // bits bit-1, bit, bit+1
            return ((1 << action) << 3) | neighbourhood;
        }

        void train(int[] before, int action, int[] after, boolean[] attended) {
            for (int c = 0; c < CHANNELS; c++) {
                if (!attended[c]) continue;
                for (int b = 0; b < BITS; b++) {
                    int idx = ((c * BITS + b) << 7) | address(before[c], b, action);
                    if (((after[c] >> b) & 1) == 1) ones[idx]++; else zeros[idx]++;
                }
            }
        }

        // Run the network forward one step. Where a neuron has no evidence, or is
        // undecided, its bit simply stays as it was.
        int[] predict(int[] state, int action, boolean[] attended) {
            int[] next = new int[CHANNELS];
            for (int c = 0; c < CHANNELS; c++) {
                if (!attended[c]) continue;
                int m = 0;
                for (int b = 0; b < BITS; b++) {
                    int idx = ((c * BITS + b) << 7) | address(state[c], b, action);
                    boolean on = ones[idx] == zeros[idx] ? ((state[c] >> b) & 1) == 1
                                                         : ones[idx] > zeros[idx];
                    if (on) m |= 1 << b;
                }
                next[c] = m;
            }
            return next;
        }

        int locationsWritten() {
            int n = 0;
            for (int i = 0; i < zeros.length; i++) if (zeros[i] + ones[i] > 0) n++;
            return n;
        }
    }

    // ------------------------------------------------------------------
    //  AXIOM 5: EMOTION (and the memory of outcomes it works from).
    //  A net of RAM neurons, each reading an 8-bit window of one channel's retina
    //  (and optionally the action), each storing a value. A neuron takes part only if
    //  its channel is attended. The net's answer is the average over taking-part neurons.
    // ------------------------------------------------------------------
    static final class AffectNet {
        static final int WINDOW = 8, STRIDE = 4;
        static final int WINDOWS = (BITS - WINDOW) / STRIDE + 1;     // 6 per channel

        private final boolean withAction;
        private final double init;
        private final double[][][] ram;                              // [channel][window][address]

        AffectNet(boolean withAction, double init) {
            this.withAction = withAction;
            this.init = init;
            int size = withAction ? (1 << (8 + 2)) : (1 << 8);
            ram = new double[CHANNELS][WINDOWS][size];
            for (double[][] ch : ram) for (double[] w : ch) Arrays.fill(w, init);
        }

        private int address(int mask, int w, int action) {
            int window = (mask >> (w * STRIDE)) & ((1 << WINDOW) - 1);
            return withAction ? (action << WINDOW) | window : window;
        }

        double read(int[] state, int action, boolean[] attended) {
            double sum = 0; int n = 0;
            for (int c = 0; c < CHANNELS; c++) {
                if (!attended[c]) continue;
                for (int w = 0; w < WINDOWS; w++) { sum += ram[c][w][address(state[c], w, action)]; n++; }
            }
            return n == 0 ? 0 : sum / n;
        }

        void write(int[] state, int action, boolean[] attended, double target) {
            double err = target - read(state, action, attended);
            for (int c = 0; c < CHANNELS; c++) {
                if (!attended[c]) continue;
                for (int w = 0; w < WINDOWS; w++) ram[c][w][address(state[c], w, action)] += RATE * err;
            }
        }

        int locationsWritten() {
            int n = 0;
            for (double[][] ch : ram) for (double[] w : ch) for (double v : w) if (v != init) n++;
            return n;
        }
    }

    // ------------------------------------------------------------------
    //  The agent
    // ------------------------------------------------------------------
    static final class Agent {
        private static final int WARMUP = 30;           // observations before attention judges
        private static final double STABLE_BELOW = 0.3; // mean change under this = worth attending to

        private final Config cfg;
        private final Random rng;
        private final NeuralStateMachine imagination = new NeuralStateMachine();
        private final AffectNet outcomes = new AffectNet(true, OPTIMISM);   // what an action tends to bring
        private final AffectNet affect = new AffectNet(false, 0.0);         // how good a state is

        // attention bookkeeping
        private final double[] meanChange = new double[CHANNELS];
        private final boolean[] attended = new boolean[CHANNELS];
        private int changeSamples = 0;
        private double[] prevRaw = null;

        // presence
        private double[] curRaw;
        private int[] prevRetina = null, curRetina = null;

        // emotion
        private double avgReward = 0, mood = 0;

        Agent(Config cfg, Random rng) { this.cfg = cfg; this.rng = rng; }

        void newEpisode() { prevRaw = null; prevRetina = null; curRetina = null; }

        int locationsWritten() {
            return imagination.locationsWritten() + outcomes.locationsWritten() + affect.locationsWritten();
        }

        // ---- AXIOMS 1 and 3: PRESENCE and ATTENTION -------------------
        void perceive(double[] raw) {
            curRaw = raw.clone();

            if (prevRaw != null) {                                   // Axiom 3: Attention
                changeSamples++;
                double rate = Math.max(1.0 / changeSamples, 0.02);
                for (int i = 0; i < raw.length; i++)
                    meanChange[i] += rate * (Math.abs(raw[i] - prevRaw[i]) - meanChange[i]);
            }
            prevRaw = raw.clone();
            for (int c = 0; c < CHANNELS; c++)
                attended[c] = !cfg.attention || changeSamples < WARMUP || meanChange[c] < STABLE_BELOW;

            prevRetina = curRetina;                                  // Axiom 1: Presence
            curRetina = encode(raw, attended);
        }

        // ---- AXIOMS 2, 4 and 5: IMAGINATION, PLANNING, EMOTION --------
        Action decide(boolean verbose) {
            Action[] acts = Action.values();
            int depth = cfg.planning ? PLAN_DEPTH : 1;
            double[] scores = new double[acts.length];
            for (int a = 0; a < acts.length; a++) scores[a] = score(curRetina, a, depth);

            double eps = cfg.emotion ? 0.05 + 0.20 * (1 - mood) / 2 : 0.15;
            boolean explore = rng.nextDouble() < eps;
            int choice;
            if (explore) {
                choice = rng.nextInt(acts.length);
            } else {
                double best = Double.NEGATIVE_INFINITY;
                for (double s : scores) best = Math.max(best, s);
                List<Integer> ties = new ArrayList<>();
                for (int a = 0; a < acts.length; a++) if (scores[a] > best - 1e-9) ties.add(a);
                choice = ties.get(rng.nextInt(ties.size()));
            }

            if (verbose) printTrace(acts, scores, depth, eps, explore, choice);
            return acts[choice];
        }

        // Worth of taking action a from a retina state, looking `depth` moves ahead.
        private double score(int[] state, int a, int depth) {
            double r = outcomes.read(state, a, attended);            // what this has brought before
            if (!cfg.imagination) return r;                          // no foresight: react to memory only

            int[] next = imagination.predict(state, a, attended);    // Axiom 2: Imagination
            double future;
            if (depth <= 1) {
                future = cfg.emotion ? affect.read(next, 0, attended) : 0;   // Axiom 5: affective value
            } else {
                future = Double.NEGATIVE_INFINITY;                   // Axiom 4: Planning
                for (int a2 = 0; a2 < Action.values().length; a2++)
                    future = Math.max(future, score(next, a2, depth - 1));
            }
            return r + GAMMA * future;
        }

        // ---- Learning from what actually happened ----------------------
        void learn(Action action, double reward, boolean ate) {
            if (cfg.emotion) {                                       // Axiom 5: mood follows reward
                avgReward += 0.05 * (reward - avgReward);
                mood = Math.tanh(8 * avgReward);
            }
            if (prevRetina == null) return;
            int a = action.ordinal();
            outcomes.write(prevRetina, a, attended, reward);
            if (cfg.imagination) imagination.train(prevRetina, a, curRetina, attended);
            if (cfg.emotion) {
                double target = ate ? reward : reward + GAMMA * affect.read(curRetina, 0, attended);
                affect.write(prevRetina, 0, attended, target);
            }
        }

        // ---- Trace output ----------------------------------------------
        private void printTrace(Action[] acts, double[] scores, int depth, double eps, boolean explore, int choice) {
            System.out.printf("  [1 Presence]    sensors = %s%n", fmt(curRaw));
            List<Integer> on = new ArrayList<>();
            for (int c = 0; c < CHANNELS; c++) if (attended[c]) on.add(c);
            System.out.printf("  [3 Attention]   attending to channels %s%n", on);
            System.out.printf("                  retina ch0 = %s%n", bar(curRetina[0]));
            if (cfg.imagination) {
                StringBuilder im = new StringBuilder();
                for (int a = 0; a < acts.length; a++) {
                    int[] n = imagination.predict(curRetina, a, attended);
                    im.append(String.format("%s -> %s   ", acts[a], bar(n[0])));
                }
                System.out.printf("  [2 Imagination] predicted ch0 after each action:%n                  %s%n",
                        im.toString().trim().replace("   ", "\n                  "));
            } else {
                System.out.printf("  [2 Imagination] (switched off)%n");
            }
            StringBuilder sc = new StringBuilder();
            for (int a = 0; a < acts.length; a++) sc.append(String.format("%s %+.2f   ", acts[a], scores[a]));
            System.out.printf("  [4 Planning]    lookahead %d:  %s%n", depth, sc.toString().trim());
            System.out.printf("  [5 Emotion]     mood %+.2f   value of this state %+.2f   explore-rate %.2f%n",
                    cfg.emotion ? mood : 0.0, cfg.emotion ? affect.read(curRetina, 0, attended) : 0.0, eps);
            System.out.printf("  => %s%s%n", acts[choice], explore ? "  (exploratory)" : "");
        }

        private static String bar(int mask) {
            StringBuilder sb = new StringBuilder();
            for (int b = 0; b < BITS; b++) sb.append(((mask >> b) & 1) == 1 ? '#' : '.');
            return sb.toString();
        }

        private static String fmt(double[] v) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < v.length; i++) sb.append(i > 0 ? " " : "").append(String.format("%+.2f", v[i]));
            return sb.append("]").toString();
        }
    }

    // ------------------------------------------------------------------
    //  Running things
    // ------------------------------------------------------------------
    static final class Session {
        final Corridor env;
        final Agent agent;

        Session(Config cfg, long seed) {
            env = new Corridor(new Random(seed));
            agent = new Agent(cfg, new Random(seed ^ 0x9E3779B97F4A7C15L));
        }

        double episode(boolean verbose, int steps) {
            env.reset();
            agent.newEpisode();
            agent.perceive(env.sense());
            double total = 0;
            for (int t = 0; t < steps; t++) {
                if (verbose) System.out.printf("%nstep %d   (food is %+d cells away)%n", t + 1, env.distanceToFood());
                Action a = agent.decide(verbose);
                double r = env.step(a);
                total += r;
                agent.perceive(env.sense());
                agent.learn(a, r, env.ate());
                if (verbose) System.out.printf("  outcome: reward %+.2f%s%n", r, env.ate() ? "  (ate!)" : "");
            }
            return total;
        }
    }

    // A scripted agent that simply walks to the food: the ceiling for this task.
    static double oracleEpisode(Random rng) {
        Corridor env = new Corridor(rng);
        env.reset();
        double total = 0;
        for (int t = 0; t < EPISODE_STEPS; t++) {
            int d = env.distanceToFood();
            total += env.step(d > 0 ? Action.RIGHT : d < 0 ? Action.LEFT : Action.EAT);
        }
        return total;
    }

    static double mean(double[] x, int from, int to) {
        double s = 0; for (int i = from; i < to; i++) s += x[i]; return s / (to - from);
    }

    static double stdErr(double[] x) {
        double m = mean(x, 0, x.length), v = 0;
        for (double d : x) v += (d - m) * (d - m);
        return Math.sqrt(v / (x.length - 1)) / Math.sqrt(x.length);
    }

    static void ablationExperiment(int seeds, int episodes) {
        final int window = 50;
        Config all = new Config();
        String[] names = { "all five axioms", "no attention", "no imagination", "no planning", "no emotion" };
        Config[] configs = {
            all, all.without("attention"), all.without("imagination"),
            all.without("planning"), all.without("emotion")
        };

        System.out.printf("Reward per %d-step episode, mean +/- standard error over %d runs of %d episodes%n",
                EPISODE_STEPS, seeds, episodes);
        System.out.printf("%-18s %-18s %-18s %s%n", "configuration", "first " + window + " episodes",
                "last " + window + " episodes", "RAM locations written");
        for (int c = 0; c < configs.length; c++) {
            final Config cfg = configs[c];
            double[] early = new double[seeds], late = new double[seeds], written = new double[seeds];
            IntStream.range(0, seeds).parallel().forEach(s -> {
                Session session = new Session(cfg, 1000 + s);
                double[] totals = new double[episodes];
                for (int e = 0; e < episodes; e++) totals[e] = session.episode(false, EPISODE_STEPS);
                early[s] = mean(totals, 0, window);
                late[s] = mean(totals, episodes - window, episodes);
                written[s] = session.agent.locationsWritten();
            });
            System.out.printf("%-18s %+6.2f +/- %-8.2f %+6.2f +/- %-8.2f %.0f%n", names[c],
                    mean(early, 0, seeds), stdErr(early), mean(late, 0, seeds), stdErr(late),
                    mean(written, 0, seeds));
        }
        double[] o = new double[2000];
        Random rng = new Random(7);
        for (int i = 0; i < o.length; i++) o[i] = oracleEpisode(rng);
        System.out.printf("%-18s %+6.2f   (scripted walk-to-food: the ceiling)%n", "reference", mean(o, 0, o.length));
    }

    static void trace(Config cfg) {
        Session session = new Session(cfg, 42);
        for (int e = 0; e < 200; e++) session.episode(false, EPISODE_STEPS);
        System.out.println("Agent trained for 200 episodes. Watching one more (first 15 steps).");
        session.episode(true, 15);
    }

    public static void main(String[] args) {
        if (args.length > 0 && args[0].equals("trace")) {
            Config cfg = new Config();
            for (String a : args) {
                switch (a) {
                    case "--no-attention"   -> cfg.attention = false;
                    case "--no-imagination" -> cfg.imagination = false;
                    case "--no-planning"    -> cfg.planning = false;
                    case "--no-emotion"     -> cfg.emotion = false;
                    default -> { }
                }
            }
            trace(cfg);
        } else {
            ablationExperiment(20, 300);
        }
    }
}
