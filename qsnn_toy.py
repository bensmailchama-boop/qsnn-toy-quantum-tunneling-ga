import random, math, argparse, statistics
from dataclasses import dataclass
import sys

# 
sys.argv = [arg for arg in sys.argv if not arg.startswith("-f")]

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---------------------------
# CONFIG
# ---------------------------
@dataclass
class Config:
    N: int = 10
    connect_p: float = 0.07
    pop_size: int = 60
    gens: int = 220
    switch_gen: int = 90
    mut_scale: float = 0.30
    thresh: float = 0.90

    # Stage 1 deceptive weights
    w_rec_stage1: float = 0.95
    w_ff_stage1: float = 0.03
    w_frag_pen_stage1: float = 0.25

    # Stage 2 penalties / density control
    rec_pen_stage2: float = 1.8
    frag_pen_stage2: float = 0.35
    sp_dead_cut: float = 0.90
    dens_max: float = 0.20
    dead_pen_w: float = 0.50
    dens_pen_w: float = 0.45

    # AND-lock thresholds
    FF_REQ: int = 10
    HUB_REQ: int = 2
    GATE_REQ: int = 12

    # Crumbs (thin staircase)
    crumb_bonus: float = 0.012

    # Evaluation noise
    eval_noise_sd: float = 0.02

    # QT knobs (stage2 only)
    protect_gens: int = 3
    tunnel_budget_frac: float = 0.15
    surprise_thresh: float = 0.06
    f_act_ceiling: float = 0.55

    a_surprise: float = 7.0
    b_disp: float = 5.0

    hbar_start: float = 1.0
    hbar_end: float = 0.1

CFG = Config()

# ---------------------------
# Genome / graph encoding
# ---------------------------
def make_edge_index(N):
    return [(i, j) for i in range(N) for j in range(N) if i != j]

EDGE_INDEX = make_edge_index(CFG.N)
GENOME_LEN = len(EDGE_INDEX)

def genome_to_matrix(g):
    N = CFG.N
    W = [[0]*N for _ in range(N)]
    for k, (i, j) in enumerate(EDGE_INDEX):
        W[i][j] = g[k]
    return W

def random_genome(rng):
    return [rng.choice([-1, 1]) if rng.random() < CFG.connect_p else 0
            for _ in range(GENOME_LEN)]

def sparsity(g):
    return 1 - sum(1 for x in g if x != 0) / GENOME_LEN

# ---------------------------
# Motifs
# ---------------------------
def count_feedforward_chains(W):
    N = CFG.N
    c = 0
    for i in range(N):
        Wi = W[i]
        for j in range(N):
            if i == j or Wi[j] == 0:
                continue
            Wj = W[j]
            if Wj[i] != 0:
                continue
            for k in range(N):
                if k == i or k == j or Wj[k] == 0:
                    continue
                if W[k][j] == 0:
                    c += 1
    return c

def count_recurrent_pairs(W):
    N = CFG.N
    c = 0
    for i in range(N):
        for j in range(i+1, N):
            if W[i][j] != 0 and W[j][i] != 0:
                c += 1
    return c

def count_inhib_hubs(W, out_thresh=3):
    N = CFG.N
    c = 0
    for i in range(N):
        out_inhib = 0
        Wi = W[i]
        for j in range(N):
            if i != j and Wi[j] == -1:
                out_inhib += 1
        if out_inhib >= out_thresh:
            c += 1
    return c

def count_fragmentation(W):
    N = CFG.N
    c = 0
    for i in range(N):
        outs = []
        Wi = W[i]
        for j in range(N):
            if i != j and Wi[j] != 0:
                outs.append(Wi[j])
        if len(outs) >= 4 and (1 in outs and -1 in outs):
            c += 1
    return c

def count_disyn_inhib_gates(W):
    N = CFG.N
    c = 0
    for j in range(N):
        exc_in = inhib_in = exc_out = 0
        for i in range(N):
            if i == j:
                continue
            v = W[i][j]
            if v == 1:
                exc_in += 1
            elif v == -1:
                inhib_in += 1
        Wj = W[j]
        for l in range(N):
            if l != j and Wj[l] == 1:
                exc_out += 1
        c += exc_in * inhib_in * exc_out
    return c

# ---------------------------
# AND-gap (basin proximity)
# ---------------------------
def and_gap(ff, hubs, gates):
    g_ff   = max(0, CFG.FF_REQ   - ff)   / CFG.FF_REQ
    g_hub  = max(0, CFG.HUB_REQ  - hubs) / max(1, CFG.HUB_REQ)
    g_gate = max(0, CFG.GATE_REQ - gates) / CFG.GATE_REQ
    return (g_ff + g_hub + g_gate) / 3.0

# ---------------------------
# Fitness
# ---------------------------
def actual_fitness(g, gen, rng):
    W = genome_to_matrix(g)
    ff = count_feedforward_chains(W)
    rec = count_recurrent_pairs(W)
    hubs = count_inhib_hubs(W)
    gates = count_disyn_inhib_gates(W)
    frag = count_fragmentation(W)
    sp = sparsity(g)
    density = 1 - sp

    if gen < CFG.switch_gen:
        rec_score = min(1.0, rec / 6)
        ff_score = min(1.0, ff / 25)
        frag_pen = min(1.0, frag / 5)
        fit = (CFG.w_rec_stage1 * rec_score +
               CFG.w_ff_stage1 * ff_score -
               CFG.w_frag_pen_stage1 * frag_pen)
    else:
        req_met = (ff >= CFG.FF_REQ and hubs >= CFG.HUB_REQ and gates >= CFG.GATE_REQ)
        rec_pen = min(1.0, rec / 10)
        frag_pen = min(1.0, frag / 5)

        if req_met:
            fit = 1.0 - CFG.rec_pen_stage2 * rec_pen - CFG.frag_pen_stage2 * frag_pen
        else:
            fit = CFG.crumb_bonus * (ff/CFG.FF_REQ +
                                     hubs/max(1, CFG.HUB_REQ) +
                                     gates/CFG.GATE_REQ) / 3.0
            fit -= CFG.rec_pen_stage2 * rec_pen * 0.3
            fit -= CFG.frag_pen_stage2 * frag_pen

    if sp > CFG.sp_dead_cut:
        fit -= CFG.dead_pen_w * (sp - CFG.sp_dead_cut) / (1 - CFG.sp_dead_cut)
    if density > CFG.dens_max:
        fit -= CFG.dens_pen_w * (density - CFG.dens_max) / (1 - CFG.dens_max)

    fit += rng.gauss(0, CFG.eval_noise_sd)
    return max(0.0, min(1.0, fit))

def predicted_meta_fitness(g):
    W = genome_to_matrix(g)
    ff = count_feedforward_chains(W)
    hubs = count_inhib_hubs(W)
    gates = count_disyn_inhib_gates(W)
    frag = count_fragmentation(W)
    sp = sparsity(g)

    ff_score   = min(1.0, ff / CFG.FF_REQ)
    hub_score  = min(1.0, hubs / max(1, CFG.HUB_REQ))
    gate_score = min(1.0, gates / CFG.GATE_REQ)
    frag_pen   = min(1.0, frag / 5)

    pred = (0.55 * ff_score +
            0.30 * hub_score +
            0.60 * gate_score -
            0.25 * frag_pen -
            0.10 * max(0, sp - 0.82) / 0.18)
    return max(0.0, min(1.0, pred))

def signature(g):
    W = genome_to_matrix(g)
    ff = count_feedforward_chains(W)
    gates = count_disyn_inhib_gates(W)
    return (min(6, ff // 5), min(6, gates // 8))

# ---------------------------
# GA ops + diversity
# ---------------------------
def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

def mean_pairwise_hamming(pop):
    n = len(pop)
    if n < 2:
        return 0.0
    s = 0
    c = 0
    for i in range(n):
        for j in range(i+1, n):
            s += hamming(pop[i], pop[j])
            c += 1
    return s / c

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tournament(pop, fits, rng, k=3):
    best = None
    bestf = -1
    for _ in range(k):
        i = rng.randrange(len(pop))
        if fits[i] > bestf:
            bestf = fits[i]
            best = pop[i]
    return best[:]

def mutate(g, rng, mut_rate):
    for i in range(len(g)):
        if rng.random() < mut_rate:
            if g[i] == 0:
                g[i] = rng.choice([-1, 1])
            else:
                g[i] = rng.choice([0, -g[i]])

def make_child(p1, p2, rng, mut_rate):
    cx = rng.randrange(1, len(p1))
    child = p1[:cx] + p2[cx:]
    mutate(child, rng, mut_rate)
    return child

# ---------------------------
# Evolution
# ---------------------------
def evolve(seed, use_qt=False):
    rng = random.Random(seed)
    mut_rate = CFG.mut_scale / GENOME_LEN

    pop = [random_genome(rng) for _ in range(CFG.pop_size)]
    protect = [0] * CFG.pop_size
    sig_success = {}
    sig_trials = {}

    best_hist = []
    hit_gen = None
    div_hist = []  # (uniq_genomes, uniq_signatures, mean_ham)

    for gen in range(CFG.gens):
        fits = [actual_fitness(ind, gen, rng) for ind in pop]

        uniq_genomes = len({tuple(ind) for ind in pop})
        uniq_sigs = len({signature(ind) for ind in pop})
        mean_ham = mean_pairwise_hamming(pop)
        div_hist.append((uniq_genomes, uniq_sigs, mean_ham))

        best = max(fits)
        best_hist.append(best)
        if hit_gen is None and gen >= CFG.switch_gen and best >= CFG.thresh:
            hit_gen = gen

        if use_qt and gen >= CFG.switch_gen:
            for i, ind in enumerate(pop):
                if protect[i] > 0 and fits[i] > 0.8:
                    s = signature(ind)
                    sig_success[s] = sig_success.get(s, 0) + 1

        hbar_eff = CFG.hbar_start - (CFG.hbar_start - CFG.hbar_end) * (gen/(CFG.gens-1))

        elite_idx = sorted(range(CFG.pop_size), key=lambda i: fits[i], reverse=True)[:2]
        new_pop = [pop[i][:] for i in elite_idx]
        new_prot = [max(0, protect[i]-1) for i in elite_idx]
        tunnels = []

        while len(new_pop) < CFG.pop_size:
            p1 = tournament(pop, fits, rng)
            p2 = tournament(pop, fits, rng)
            child = make_child(p1, p2, rng, mut_rate)

            if (not use_qt) or (gen < CFG.switch_gen):
                new_pop.append(child); new_prot.append(0); continue

            f_act = actual_fitness(child, gen, rng)
            f_pred = predicted_meta_fitness(child)
            surprise = f_pred - f_act

            Wc = genome_to_matrix(child)
            ff_c = count_feedforward_chains(Wc)
            hubs_c = count_inhib_hubs(Wc)
            gates_c = count_disyn_inhib_gates(Wc)
            gap_c = and_gap(ff_c, hubs_c, gates_c)

            disp = hamming(child, p1) / GENOME_LEN
            disp_score = disp * (1 + 0.8 * sparsity(p1))

            s = signature(child)
            sig_trials[s] = sig_trials.get(s, 1) + 1
            success_rate = sig_success.get(s, 0) / sig_trials[s]
            interference = 0.5 + success_rate

            p_tunnel = sigmoid(
                CFG.a_surprise * surprise +
                CFG.b_disp * disp_score -
                2.0 * gap_c
            ) * hbar_eff * interference
            p_tunnel = max(0.0, min(1.0, p_tunnel))

            if (gap_c < 0.9) and (surprise > CFG.surprise_thresh) and \
               (f_act < CFG.f_act_ceiling) and (rng.random() < p_tunnel):
                tunnels.append((p_tunnel, child))
            else:
                new_pop.append(child); new_prot.append(0)

        if use_qt and gen >= CFG.switch_gen and tunnels:
            budget = max(1, int(CFG.pop_size * CFG.tunnel_budget_frac))
            tunnels.sort(key=lambda x:x[0], reverse=True)
            for _, child in tunnels[:budget]:
                new_pop.append(child)
                new_prot.append(CFG.protect_gens)

        pop, protect = new_pop[:CFG.pop_size], new_prot[:CFG.pop_size]

    return best_hist, hit_gen, div_hist

def run_many(seeds, use_qt=False):
    hists=[]; hits=[]; divs=[]
    for s in seeds:
        hist, hit, div_hist = evolve(s, use_qt=use_qt)
        hists.append(hist); hits.append(hit); divs.append(div_hist)
    return hists, hits, divs

def success_stats(hits):
    succ=[h for h in hits if h is not None]
    rate=len(succ)/len(hits)
    med=statistics.median(succ) if succ else None
    mean=statistics.mean(succ) if succ else None
    return rate, med, mean

# ---------------------------
# Main + plotting 
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--plot", action="store_true")

    # Parse known args; 
    args, _ = ap.parse_known_args()

    # Auto-plot
    in_notebook = ("ipykernel" in sys.modules)
    if in_notebook and not args.plot:
        args.plot = True

    seeds_base=[1000+i for i in range(args.seeds)]
    seeds_qt = seeds_base  # paired seeds

    base_hists, base_hits, base_divs = run_many(seeds_base, use_qt=False)
    qt_hists, qt_hits, qt_divs       = run_many(seeds_qt, use_qt=True)

    gens=len(base_hists[0])
    mean_base=[statistics.mean(h[g] for h in base_hists) for g in range(gens)]
    mean_qt  =[statistics.mean(h[g] for h in qt_hists) for g in range(gens)]

    def mean_div(divs, idx):
        return [statistics.mean(d[g][idx] for d in divs) for g in range(gens)]

    mean_uniq_base = mean_div(base_divs, 0)
    mean_sigs_base = mean_div(base_divs, 1)
    mean_ham_base  = mean_div(base_divs, 2)

    mean_uniq_qt = mean_div(qt_divs, 0)
    mean_sigs_qt = mean_div(qt_divs, 1)
    mean_ham_qt  = mean_div(qt_divs, 2)

    base_rate, base_med, base_mean = success_stats(base_hits)
    qt_rate, qt_med, qt_mean       = success_stats(qt_hits)

    print("=== Task B AND-locked stage2 ===")
    print(f"Seeds per condition: {args.seeds}")
    print(f"Stage switch gen: {CFG.switch_gen}")
    print("--- Success stats (hit gen >= switch and best>=thresh) ---")
    print(f"GA     success_rate={base_rate:.2f}, median_hit={base_med}, mean_hit={base_mean}")
    print(f"GA+QT  success_rate={qt_rate:.2f}, median_hit={qt_med}, mean_hit={qt_mean}")
    print("Example hit gens:")
    print(" GA   :", base_hits[:10])
    print(" GA+QT:", qt_hits[:10])

    if not args.plot or plt is None:
        return

    # fitness plot
    plt.figure(figsize=(8,5))
    plt.plot(range(gens), mean_base, label="Mean best (GA)")
    plt.plot(range(gens), mean_qt,   label="Mean best (GA+QT)")
    plt.axvline(CFG.switch_gen, linestyle="--", label="Switch gen")
    plt.xlabel("Generation"); plt.ylabel("Best fitness")
    plt.title("Fitness: GA vs GA+QT")
    plt.legend(); plt.tight_layout(); plt.show()

    # diversity plots
    plt.figure(figsize=(8,5))
    plt.plot(range(gens), mean_uniq_base, label="Unique genomes (GA)")
    plt.plot(range(gens), mean_uniq_qt,   label="Unique genomes (GA+QT)")
    plt.axvline(CFG.switch_gen, linestyle="--")
    plt.xlabel("Generation"); plt.ylabel("Count")
    plt.title("Population uniqueness")
    plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(range(gens), mean_sigs_base, label="Unique signatures (GA)")
    plt.plot(range(gens), mean_sigs_qt,   label="Unique signatures (GA+QT)")
    plt.axvline(CFG.switch_gen, linestyle="--")
    plt.xlabel("Generation"); plt.ylabel("Count")
    plt.title("Signature diversity")
    plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(range(gens), mean_ham_base, label="Mean Hamming (GA)")
    plt.plot(range(gens), mean_ham_qt,   label="Mean Hamming (GA+QT)")
    plt.axvline(CFG.switch_gen, linestyle="--")
    plt.xlabel("Generation"); plt.ylabel("Mean distance")
    plt.title("Mean pairwise Hamming distance")
    plt.legend(); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
