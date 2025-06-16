# 🤖 Reinforcement Learning Project

This repository contains the implementation of a Reinforcement Learning (RL) project developed as part of the RL course. The goal is to train and compare multiple RL agents on two distinct environments from the [Gymnasium](https://gymnasium.farama.org/) library, focusing on performance, convergence, and the trade-off between exploration and exploitation.

---

## 📌 Objectives

- Solve **two different Gymnasium environments** using different RL algorithms.
- Compare the performance of algorithms such as Q-Learning, SARSA, Monte Carlo, and PPO.
- Visualize learning progress with reward curves, success rates, and convergence plots.
- Analyze agent behavior and interpret learned policies.

---

## 🧱 Project Structure

```
RL_GP/
|── notebooks/
|── src/
│   └── agents/             # RL algorithm implementations (Q-Learning, SARSA, Monte Carlo, etc.)
│   └── environments/       # Custom wrappers and environment configurations (Blackjack, Pendulum)
│   └── training/           # Training scripts and grid search utilities
│   └── evaluation/         # Evaluation scripts and comparative analysis
│   └── main.py             # Main entry point to run all experiments
├── config/                 # YAML configuration files for experiments
├── output/                 # Generated plots, HTML reports, and metrics for each experiment
│   └── analysis/       
│       ├── montecarlo/
│       ├── sarsa/
│       ├── qlearning/
│       ├── pendulum_qlearning/
│       └── pendulum_sarsa/
|   └── checkpoints/
|   └── best_qtable.npy
├── logs/                   # Training logs for each experiment
├── report/                 # Final HTML report and supporting files
├── LICENSE
├── requirements.txt        # Python dependencies
└── README.md               # This file

```

---

## ⚙️ Installation

1. Clone this repository:

```bash
git clone https://github.com/TiagoPedrotkd/RL_GP.git
cd RL_GP
```

2. *(Optional)* Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Run All Experiments (Blackjack and Pendulum)

```bash
python -m src.main
```

This will:
- Train all agents (Monte Carlo, SARSA, Q-Learning, Random) on Blackjack.
- Train Q-Learning and SARSA agents on Pendulum.
- Perform grid search for each agent/policy.
- Save all logs, plots, and HTML reports in `output/analysis/` and `logs/`.

### Run Only Blackjack Experiments

```bash
python -m src.training.train_blackjack
```

### Run Only Pendulum Experiments

```bash
python -m src.training.train_pendulum
```

> All training metrics, visualizations, and HTML reports will be saved in the `output/analysis/` directory, organized by agent and policy.

---

## 📊 Visualizations & Reports

- **Learning curves** (moving average of returns)
- **Return histograms**
- **Q-table heatmaps** and **policy visualizations**
- **Comparative plots** across policies and agents
- **HTML reports** for each experiment and a global comparative report in `report/report.html`

---

## 📘 Report

A detailed description of the methodology, evaluation metrics, challenges, and results is available in [`report/report.html`](./report/report.html) and [`report.pdf`](./report.pdf).

---

## 👥 Authors

Developed by:

- Tiago Pedro (tiagopedrosoares02@gmail.com)
- Tomás Silva (tomasestrociosilva@gmail.com)
- Cadmo Diogo (cadmaya@hotmail.com)

---

## 📄 License

This project is for educational purposes only. All rights reserved by the authors.
