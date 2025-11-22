# Q-SNN Toy: Quantum-Tunneling-Inspired Genetic Algorithm

This repository contains a small **proof-of-concept demo** showing how a quantum-tunneling-inspired meta-search heuristic can help a genetic algorithm (GA) escape deceptive basins when evolving sparse signed graphs.

The long-term idea:  
**Quantum-inspired search (outer loop) × Classical GA (inner loop) is a better exploration in neuroevolution.**

This toy model is the first public milestone toward **Q-SNN**.

---

## What this toy does

- Evolves signed directed graphs (size N=10)
- Uses a deceptive two-stage fitness landscape
- Compares:
  - **GA baseline**
  - **GA + QT (tunneling / surprise-driven filtering)**

After the landscape switches to an AND-locked target:
- predicted meta-fitness + surprise score guide tunneling probability  
- motif signatures accumulate “interference” weight  
- high-surprise, low-fitness children can *tunnel* into the population  
- protects diversity for a few generations  

All lightweight and fully CPU-only.

---

## Why quantum-inspired?

The idea is not physical quantum simulation.  
It’s *borrowing quantum metaphors*:

- **Surprise = interference**  
- **Dispersion = wavefunction spread**  
- **Tunneling probability** ≈ escape from deceptive basins  
- **hbar annealing** = exploration-to-exploitation schedule  

This is intentionally minimal — a toy to show the concept works.

---

## Usage

### Run with paired seeds (GA vs GA+QT)
```bash
python qsnn_toy.py --seeds 30
