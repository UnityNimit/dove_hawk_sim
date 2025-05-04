# Hawk-Dove Evolutionary Simulation 🦅🕊️  
**A Python simulation of evolutionary game theory with GUI and statistical analysis**  

---

## 📌 Table of Contents
- [Problem Statement](#-problem-statement)
- [Introduction](#-introduction)
- [Installation](#-installation)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [Results & Analysis](#-results--analysis)
- [Conclusion](#-conclusion)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Problem Statement
This project simulates the **Hawk-Dove conflict model** to answer:  
*"How do aggressive (Hawk) and cooperative (Dove) strategies evolve in populations competing for limited resources?"*  

**Key Objectives**:
1. Model population dynamics under varying resource conditions.
2. Quantify survival rates and equilibrium states.
3. Analyze trade-offs between aggression and cooperation.

**Real-World Applications**:
- Animal behavior studies (territorial disputes)
- Economic strategy optimization
- Ecosystem resource management

---

## 📖 Introduction
The Hawk-Dove model, rooted in evolutionary game theory, explores how conflict strategies stabilize in biological populations. Agents compete for food pairs, with survival and reproduction determined by probabilistic rules:  
- **Hawks** risk starvation in fights but gain more food in uneven conflicts.  
- **Doves** share resources safely but lose opportunities to hawks.  

**Key Concepts**:  
- **Nash Equilibrium**: Stable strategy mix where no agent benefits by changing tactics.  
- **Payoff Matrix**:  
  | Encounter    | Hawk       | Dove       |  
  |--------------|------------|------------|  
  | **Hawk**     | 0 food     | 1.5 food   |  
  | **Dove**     | 0.5 food   | 1 food     |  

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- `pip` package manager

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/hawk-dove-simulation.git
   cd hawk-dove-simulation
   Install dependencies:

bash
pip install -r requirements.txt
# or manually:
pip install matplotlib scipy tkinter
Run the simulation:

bash
python main.py

🎮 Usage
Set Parameters:

Initial Hawks/Doves (e.g., 50 each)

Food pairs per day (e.g., 50)

Hawk-Hawk payoff (default: 0.0)

Generations (e.g., 100)

Run Simulation:

Click "Run Simulation" to start.

Real-time population graphs and statistics update automatically.

Analyze Results:

Export graphs as PNG/PDF.

Compare survival rates and encounter frequencies.

🖼️ Screenshots
Input Interface	Population Trends	Statistics Dashboard
GUI Input	Population Graph	Stats Dashboard
📊 Results & Analysis
Population Dynamics
Dove Dominance: Avg 199.43 doves vs. 160.11 hawks.

Equilibrium: ~55% doves, ~45% hawks (mixed Nash equilibrium).

Survival Rates
Strategy	Survival Rate
Hawk	67.93%
Dove	75.58%
Encounter Statistics (100 Runs)
Type	Count	Outcome
Hawk-Hawk	2,721	0 food (energy depletion)
Hawk-Dove	5,495	Hawk gains 1.5, Dove 0.5
Dove-Dove	4,247	Stable 1 food each
🏁 Conclusion
Risk vs. Reward: Doves thrive due to consistent survival, while hawks face energy-draining conflicts.

Equilibrium: Mixed populations persist, demonstrating evolutionary stability.

Extensions: Future work could add spatial dynamics or probabilistic strategies.

🤝 Contributing
Fork the repository.

Create a branch: git checkout -b feature/new-feature.

Commit changes: git commit -m "Add feature".

Push: git push origin feature/new-feature.

Submit a Pull Request.

Suggested Improvements:

Add territorial behavior (grid-based movement)

Implement mutation rates for strategy switching
