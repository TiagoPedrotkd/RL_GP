
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
├── agents/             # RL algorithm implementations
├── environments/       # Custom wrappers and environment configurations
├── training/           # Training scripts for each agent/environment
├── evaluation/         # Evaluation scripts and comparative analysis
├── results/            # Training logs, reward curves, and saved models
├── notebooks/          # Jupyter notebooks for analysis and plotting
├── config/             # YAML configuration files
├── requirements.txt    # Python dependencies
├── report.pdf          # Final project report
└── README.md           # This file
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

### Train Agent on Environment 1:

```bash
python training/train_env1.py
```

### Evaluate Agent on Environment 1:

```bash
python evaluation/evaluate_env1.py
```

> Training metrics and visualizations will be saved in the `results/` directory.

---

## 📊 Visualizations

- Total reward per episode
- Moving average of rewards
- Convergence curves
- Performance comparison across agents

> Visual outputs and further analysis can be found in the `notebooks/` folder.

---

## 📘 Report

A detailed description of the methodology, evaluation metrics, challenges, and results is available in [`report.pdf`](./report.pdf).

---

## 👥 Authors

Developed by:

- Name 1 (email)
- Name 2 (email)
- Name 3 (email)

---

## 📄 License

This project is for educational purposes only. All rights reserved by the authors.
