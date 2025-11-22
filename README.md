# Q-SNN Toy: Quantum-Tunneling-Inspired Genetic Algorithm

This repository contains a small **proof-of-concept demo** showing how a quantum-tunneling-inspired meta-search heuristic can help a genetic algorithm (GA) escape deceptive basins when evolving sparse signed graphs.

This is the first public step toward my larger project **Q-SNN**  
(Quantum-Inspired Search for Spiking Neural Network Evolution).

---

## Overview

This toy model:

- Evolves a population of signed directed graphs (N=10 nodes)
- Uses a **two-stage deceptive fitness landscape**
- Compares:
  - baseline **GA**
  - **GA + QT** (surprise-driven tunneling + motif interference)

After the landscape switches to an AND-locked target, the QT layer:

- predicts meta-fitness from motif signatures  
- computes a **surprise score**  
- uses surprise + dispersion + gap penalties to assign a **tunneling probability**  
- reinforces successful motif signatures (“interference”)  
- protects tunneled individuals for a few generations  

This helps the GA explore deeper and escape deceptive local minima.

---

## Why “Quantum-Inspired”?

This toy is NOT quantum computation.  
It borrows **quantum metaphors** for smarter exploration:

- **Surprise** ≈ constructive interference  
- **Dispersion** ≈ wave spread / search breadth  
- **Tunneling probability** ≈ escape route through deceptive basins  
- **hbar annealing** ≈ exploration → exploitation schedule  

Minimal implementation — but surprisingly effective.

---

##  How to Run

### Default demo (30 paired seeds)
python qsnn_toy.py --seeds 30

Force plotting (if not in notebook)
python qsnn_toy.py --seeds 30 --plot

This will print:
success rate
median hit generation
example hits

And will generate plots for:
mean best fitness
unique genomes
unique motif signatures
mean Hamming distance

## Example Outputs
<img width="790" height="490" alt="17638352003066470759592365644120" src="https://github.com/user-attachments/assets/172d7b36-2d9a-4b6a-a1cb-8b50b7b6c2a3" />
<img width="790" height="490" alt="17638355671095785366008643126605" src="https://github.com/user-attachments/assets/7f4c2a83-73ca-42d7-894c-e74b41d859b1" />
<img width="790" height="490" alt="17638382222886530622956380070999" src="https://github.com/user-attachments/assets/02fa6c4f-4f04-4f3d-81b1-e9f1dedee633" />
<img width="790" height="490" alt="17638382759695089585922434739659" src="https://github.com/user-attachments/assets/d73a2c62-0030-48b1-ad99-624b81ac997e" />

## Key Parameters (Config)

All tunable parameters live in the Config dataclass:

Parameter	Meaning

switch_gen	When the landscape becomes AND-locked
thresh	Fitness threshold for success
w_rec_stage1, w_ff_stage1	Stage 1 deceptive weights
rec_pen_stage2, frag_pen_stage2	Stage 2 penalties
crumb_bonus	Small reward when close to target motifs
eval_noise_sd	Evaluation noise
tunnel_budget_frac	Max % of tunneled individuals per generation
surprise_thresh	Minimum meta-surprise to consider tunneling
hbar_start, hbar_end	Annealing schedule for tunneling intensity
protect_gens	Number of gens protected after tunneling
a_surprise, b_disp	Weighting of surprise and dispersion

## Motifs Used
Feedforward chains
Recurrent pairs
Inhibitory hubs
Disynaptic inhibitory gates
Fragmentation pattern
These form the motif-based predicted meta-fitness.


---

## Roadmap (near future)

[ ] Integrate QT with LIF/AdEx controllers

[ ] Test on animat tasks

[ ] Add ablation studies

[ ] Compare with CMA-ES + Novelty Search

[ ] Q-SNN preprint (2026)



---

## License

Released under the MIT License.

## ======Author=========
Chama Bensmail
Computational Neuroscience • Neuroevolution • Explainability •
Quantum-Inspired Search • Spiking Neural Networks
