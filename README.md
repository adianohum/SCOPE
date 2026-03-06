# SCOPE: Context-Aware Quantum Security for Federated Learning

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Powered%20by-PyTorch-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of the **SCOPE** (Scalable Context-aware Protection Environment) framework. 

## 📌 Overview

Uniform application of Quantum Key Distribution (QKD) in Federated Learning assumes abundant cryptographic resources; however, real deployments face finite key-generation rates and heterogeneous client contributions, making security overprovisioning a systemic inefficiency. 

**SCOPE** is a scalable, context-aware protection framework that models quantum keys as a constrained resource. It dynamically allocates security according to client impact, key availability, and fairness. By treating quantum security as a control variable, SCOPE reduces cryptographic consumption by over 3x, virtually eliminates key starvation under severe scarcity, and preserves model convergence and robustness.

## 🚀 Repository Structure

```text
SCOPE/
├── bazooka_qkd_fl_full.py  # Main simulation framework (FL, QKD, Networking)
├── LICENSE                 # MIT License
└── README.md               # This documentation

## 🛠️ Installation & Setup
The framework is built in Python and requires standard machine learning libraries.

# Clone the repository
git clone [https://github.com/adianohum/SCOPE.git](https://github.com/adianohum/SCOPE.git)
cd SCOPE

# Install dependencies
pip install torch torchvision numpy pandas matplotlib

## 💻 Usage
The entire experimental framework—including baselines, the proposed RIGHT_SIZED policy, Federated Learning loops, and network simulation—is contained within bazooka_qkd_fl_full.py.

1. Run the Full Simulation Grid
You can run the full grid of experiments across different seeds and datasets (e.g., MNIST, CIFAR-10) using multiprocessing.

# Example for MNIST with 5 seeds using 4 parallel jobs
python bazooka_qkd_fl_full.py run-grid --out results --dataset mnist --n-seeds 5 --jobs 4

2. Summarize Results and Generate Figures
Once the grid execution is complete, use the summarize command to automatically generate LaTeX tables, CSV summaries, and performance figures (CDFs, boxplots, learning curves).

python bazooka_qkd_fl_full.py summarize --in results --out results_summary

## 🔗 Citation
If you use this code in your research, please cite our paper:

@inproceedings{Maia2026scope,
  title={SCOPE: Segurança Quântica Sensível ao Contexto para Aprendizado Federado, Além do Superdimensionamento na Alocação de Chaves QKD},
  author={Adriano Maia and Isys Sant'Anna and Marcus Freire and Thiago Mello and Bruno Tardiole and Bruno Guazzelli and Dionisio Leite and Maycon Peixoto},
  booktitle={Anais do Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos (SBRC)},
  year={2026}
}

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
This work was partially supported by the QuIIN project "Integração CV-QKD com Redes Clássicas", supported by QuIIN - Quantum Industrial Innovation, Centro de Competência EMBRAPII CIMATEC. It was also supported by CAPES (Finance Code 001) and CNPq (Grant nº 403231/2023-0).
