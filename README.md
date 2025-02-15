# Advanced Hawk-Dove Evolutionary Simulator 🦅🕊️

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-brightgreen.svg)](https://www.python.org/)
[![Built with Tkinter](https://img.shields.io/badge/Built%20with-Tkinter-red.svg)](https://docs.python.org/3/library/tkinter.html)

An interactive desktop application for simulating the Hawk-Dove evolutionary game theory model. This tool provides a rich GUI to configure, run, and analyze population dynamics with real-time plotting, batch simulations, and built-in statistical hypothesis testing.

---

## 🚀 Live Demo & Tutorial

Watch a quick demonstration of the simulator's features, from running a single simulation to performing an A/B hypothesis test.

![Simulator Demo GIF](https://github.com/your-username/your-repo-name/blob/main/media/simulation_demo.gif)
*(To make this GIF: Record your screen while using the app, save it as `simulation_demo.gif`, place it in a `media` folder, and update the link!)*

### 🎥 Full Video Tutorial
For a detailed walkthrough of all features and the science behind the model, watch our tutorial on YouTube:
[<img src="https://img.youtube.com/vi/YOUTUBE_VIDEO_ID/hqdefault.jpg" width="400">](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID)
*(Replace `YOUTUBE_VIDEO_ID` with your video's ID)*

---

## ✨ Key Features

*   **Interactive Simulation:** Configure initial populations, resources, and payoff values.
*   **Real-time Visualization:** Watch hawk and dove populations evolve over generations with dynamic Matplotlib graphs.
*   **In-depth Statistics:** Get instant analysis on survival rates, population means, encounter types, and food distribution.
*   **Distribution Analysis:** Run batch simulations to generate and visualize the distribution of final population counts.
*   **A/B Hypothesis Testing:** Compare two different scenarios (e.g., high vs. low resources) and use built-in T-tests and Chi-squared tests to determine if the outcomes are statistically significant.
*   **Modern GUI:** A clean, tabbed interface built with `ttk` for a smooth user experience.
*   **No Installation Needed:** Download and run the standalone `.exe` on Windows.

---

## 📥 Installation & Usage

### For Users (The Easy Way)
1.  Go to the [**Releases Page**](https://github.com/your-username/your-repo-name/releases) of this repository.
2.  Download the `HawkDoveSim.zip` file from the latest release.
3.  Unzip the file and run `HawkDoveSim.exe`. No installation required!

### For Developers (From Source)
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Set up a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application**:
    ```bash
    python main.py
    ```
---

## 🔬 How It Works

This project simulates the classic Hawk-Dove conflict model. Agents compete for a limited number of food resources. Their success determines their survival and reproduction.

*   **Population:** A mix of "Hawks" (aggressive strategy) and "Doves" (cooperative strategy).
*   **Resources:** Agents are randomly assigned to a set number of food resources.
*   **Encounters:**
    *   **One Agent:** If an agent is alone at a resource, it gets 2 food units and reproduces.
    *   **Two Agents:** If two agents arrive at the same resource, a conflict occurs based on the **Payoff Matrix**.

The simulation's default payoff matrix (where the value of the resource is 2 food units):

| Opponent → <br> Player ↓ | Hawk (Aggressive) | Dove (Cooperative) |
| :--- | :--- | :--- |
| **Hawk** | **0** food each <br> (Cost of fighting is high) | **2** food for Hawk <br> **0** for Dove |
| **Dove** | **0** food for Dove <br> **2** for Hawk | **1** food each <br> (They share the resource) |

*   **Survival & Reproduction:**
    *   **< 1 food:** The agent starves and is removed.
    *   **>= 1 food:** The agent survives to the next generation.
    *   **>= 2 food:** The agent survives *and* produces one offspring of the same type.

This simple set of rules leads to complex, emergent population dynamics, often stabilizing at a mixed equilibrium where both strategies coexist.

---

## 🤝 Contributing

Contributions are welcome! Whether it's a new feature, a bug fix, or documentation improvements, please feel free to contribute.

1.  Fork the repository.
<!-- Formatting improvements for better readability -->
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a **Pull Request**.

### Potential Future Improvements
*   Add spatial dynamics (agents on a grid).
*   Implement mutation (a hawk's offspring can be a dove, and vice-versa).
*   Add more complex strategies (e.g., "Retaliator" or "Bully").

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### 📚 References
1.  Maynard Smith, J. (1982). *Evolution and the Theory of Games*.
2.  Nowak, M. A. (2006). *Evolutionary Dynamics: Exploring the Equations of Life*.
<!-- Simulated change for commit history -->
