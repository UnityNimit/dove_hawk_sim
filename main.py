# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import statistics
from scipy.stats import t

class SimulationStats:
    def __init__(self):
        self.hawk_populations = []
        self.dove_populations = []
        self.hawk_survival_rates = []
        self.dove_survival_rates = []
        self.encounter_counts = {'HH': 0, 'HD': 0, 'DD': 0}
        self.food_distribution = []

    def update_stats(self, hawks, doves, survival_data, encounters, food_data):
        self.hawk_populations.append(hawks)
        self.dove_populations.append(doves)
        self.hawk_survival_rates.append(survival_data['hawk'])
        self.dove_survival_rates.append(survival_data['dove'])
        for key in self.encounter_counts:
            self.encounter_counts[key] += encounters[key]
        self.food_distribution.extend(food_data)

    def get_metrics(self):
        def mean_ci(data, confidence=0.95):
            if len(data) < 2:
                return (0, 0, 0)
            mean = statistics.mean(data)
            std_err = statistics.stdev(data) / (len(data)**0.5)
            dof = len(data) - 1
            t_crit = t.ppf((1 + confidence)/2, dof)
            return (mean, mean - t_crit*std_err, mean + t_crit*std_err)
            
        return {
            'hawk_mean': mean_ci(self.hawk_populations),
            'dove_mean': mean_ci(self.dove_populations),
            'hawk_survival': statistics.mean(self.hawk_survival_rates) if self.hawk_survival_rates else 0,
            'dove_survival': statistics.mean(self.dove_survival_rates) if self.dove_survival_rates else 0,
            'encounters': self.encounter_counts,
            'food_stats': (min(self.food_distribution), statistics.mean(self.food_distribution), max(self.food_distribution))
        }

def simulate_hawk_dove(
    initial_hawks=50,
    initial_doves=50,
    generations=100,
    num_food_pairs=50,
    hawk_hawk_payoff=0.0,
    max_population=None
):
    stats = SimulationStats()
    population = ["hawk"] * initial_hawks + ["dove"] * initial_doves
    random.shuffle(population)

    hawk_history = []
    dove_history = []
    total_history = []

    for generation in range(generations):
        current_pop = len(population)
        hawk_count = population.count("hawk")
        dove_count = population.count("dove")

        hawk_history.append(hawk_count)
        dove_history.append(dove_count)
        total_history.append(current_pop)

        if current_pop == 0:
            remaining = generations - generation
            hawk_history.extend([0]*remaining)
            dove_history.extend([0]*remaining)
            total_history.extend([0]*remaining)
            break

        # Daily food distribution
        food_locations = {i: [] for i in range(num_food_pairs)}
        agent_food = {i: 0 for i in range(current_pop)}
        survival_data = {'hawk': 0, 'dove': 0}
        encounter_data = {'HH': 0, 'HD': 0, 'DD': 0}
        food_values = []

        for agent_idx in range(current_pop):
            chosen_pair = random.randrange(num_food_pairs)
            food_locations[chosen_pair].append(agent_idx)

        for pair, agents in food_locations.items():
            num_agents = len(agents)
            types = [population[idx] for idx in agents]
            
            if num_agents == 1:
                agent_food[agents[0]] = 2.0
            elif num_agents == 2:
                t1, t2 = types
                if t1 == "dove" and t2 == "dove":
                    agent_food[agents[0]] = 1.0
                    agent_food[agents[1]] = 1.0
                    encounter_data['DD'] += 1
                elif {t1, t2} == {"hawk", "dove"}:
                    agent_food[agents[0]] = 1.5 if t1 == "hawk" else 0.5
                    agent_food[agents[1]] = 1.5 if t2 == "hawk" else 0.5
                    encounter_data['HD'] += 1
                else:
                    agent_food[agents[0]] = hawk_hawk_payoff
                    agent_food[agents[1]] = hawk_hawk_payoff
                    encounter_data['HH'] += 1
            elif num_agents > 2:
                for agent in agents:
                    agent_food[agent] = 0.0

        food_values = list(agent_food.values())
        new_population = []
        type_counts = {'hawk': 0, 'dove': 0}
        
        for idx in range(current_pop):
            food = agent_food[idx]
            agent_type = population[idx]
            
            survives = False
            if food >= 1.5:
                survives = True
                if random.random() < 0.5:
                    new_population.append(agent_type)
            elif food >= 1.0:
                survives = True
            elif food >= 0.5:
                survives = random.random() < 0.5
                
            if survives:
                new_population.append(agent_type)
                type_counts[agent_type] += 1

        survival_rates = {
            'hawk': type_counts['hawk']/hawk_count if hawk_count > 0 else 0,
            'dove': type_counts['dove']/dove_count if dove_count > 0 else 0
        }
        
        stats.update_stats(hawk_count, dove_count, survival_rates, encounter_data, food_values)
        population = new_population
        if max_population and len(population) > max_population:
            population = random.sample(population, max_population)

    hawk_history.append(population.count("hawk"))
    dove_history.append(population.count("dove"))
    total_history.append(len(population))

    return hawk_history, dove_history, total_history, stats.get_metrics()

class HawkDoveApp:
    def __init__(self, root):
        self.root = root
        self.fig = None
        self.ax = None
        self.canvas = None
        self.result_queue = queue.Queue()
        self.polling_after_id = None
        self.stats_labels = {}
        
        self.configure_styles()
        self.create_widgets()
        self.plot_results([], [], [], {})

    def configure_styles(self):
        self.root.title("Hawk-Dove Simulator with Stats Dashboard")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2D3748")

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TFrame", background="#2D3748")
        self.style.configure("TLabel", foreground="#CBD5E0", background="#2D3748")
        self.style.configure("Header.TLabel", font=('Segoe UI', 12, 'bold'))
        self.style.configure("TEntry", fieldbackground="#E2E8F0")
        self.style.configure("TButton", 
                           background="#4299E1", 
                           foreground="black", 
                           font=('Segoe UI', 10, 'bold'),
                           borderwidth=0)

    def create_widgets(self):
        # Input Frame
        input_frame = ttk.Frame(self.root)
        input_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.Y)

        # Population Inputs
        ttk.Label(input_frame, text="Initial Hawks:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.hawks_entry = ttk.Entry(input_frame, width=15)
        self.hawks_entry.insert(0, "50")
        self.hawks_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Initial Doves:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.doves_entry = ttk.Entry(input_frame, width=15)
        self.doves_entry.insert(0, "50")
        self.doves_entry.grid(row=1, column=1, padx=5, pady=5)

        # Simulation Parameters
        ttk.Label(input_frame, text="Food Pairs/Day:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.food_entry = ttk.Entry(input_frame, width=15)
        self.food_entry.insert(0, "50")
        self.food_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Hawk-Hawk Payoff:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.payoff_entry = ttk.Entry(input_frame, width=15)
        self.payoff_entry.insert(0, "0.0")
        self.payoff_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Generations:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.generations_entry = ttk.Entry(input_frame, width=15)
        self.generations_entry.insert(0, "100")
        self.generations_entry.grid(row=4, column=1, padx=5, pady=5)

        # Control Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=15)

        self.run_button = ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(side=tk.LEFT, padx=10)

        self.status_label = ttk.Label(button_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Statistics Frame
        self.create_stats_frame()

    def create_stats_frame(self):
        stats_frame = ttk.Frame(self.root)
        stats_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH, expand=True)

        ttk.Label(stats_frame, text="Statistics Dashboard", style='Header.TLabel').pack(pady=10)
        
        metrics = [
            ('Hawk Population', 'hawk_mean'),
            ('Dove Population', 'dove_mean'),
            ('Hawk Survival Rate', 'hawk_survival'),
            ('Dove Survival Rate', 'dove_survival'),
            ('Min Food', 'food_stats.0'),
            ('Avg Food', 'food_stats.1'),
            ('Max Food', 'food_stats.2'),
            ('H-H Encounters', 'encounters.HH'),
            ('H-D Encounters', 'encounters.HD'),
            ('D-D Encounters', 'encounters.DD')
        ]

        for label, key in metrics:
            row = ttk.Frame(stats_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=20, anchor='w').pack(side=tk.LEFT)
            lbl = ttk.Label(row, text="0.00", width=20)
            lbl.pack(side=tk.RIGHT)
            self.stats_labels[key] = lbl

    def update_stats_dashboard(self, metrics):
        for key, lbl in self.stats_labels.items():
            try:
                parts = key.split('.')
                value = metrics
                for part in parts:
                    if part.isdigit():
                        value = value[int(part)]
                    else:
                        value = value[part]
                
                if isinstance(value, tuple):
                    text = f"{value[0]:.2f} ({value[1]:.2f}-{value[2]:.2f})"
                elif isinstance(value, float):
                    text = f"{value:.2%}"
                else:
                    text = f"{value}"
                lbl.config(text=text)
            except (KeyError, TypeError):
                lbl.config(text="N/A")

    def plot_results(self, hawks, doves, totals, metrics):
        if not self.fig:
            self.fig, self.ax = plt.subplots(figsize=(8, 6), facecolor="#2D3748")
            self.ax.set_facecolor("#2D3748")
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        else:
            self.ax.clear()

        generations = range(len(hawks))
        self.ax.plot(generations, hawks, label="Hawks", color="#FF4500")
        self.ax.plot(generations, doves, label="Doves", color="#228B22")
        self.ax.plot(generations, totals, label="Total", color="#CBD5E0", linestyle=':')
        
        self.ax.set_xlabel("Generation", color="#CBD5E0")
        self.ax.set_ylabel("Population", color="#CBD5E0")
        self.ax.set_title("Population Dynamics", color="#F7FAFC")
        self.ax.set_ylim(bottom=0)
        
        legend = self.ax.legend(facecolor='#4A5568')
        for text in legend.get_texts():
            text.set_color("#F7FAFC")
        
        self.fig.tight_layout()
        self.canvas.draw()
        self.update_stats_dashboard(metrics)

    def run_simulation(self):
        if self.polling_after_id:
            self.root.after_cancel(self.polling_after_id)
        
        try:
            params = {
                "initial_hawks": int(self.hawks_entry.get()),
                "initial_doves": int(self.doves_entry.get()),
                "generations": int(self.generations_entry.get()),
                "num_food_pairs": int(self.food_entry.get()),
                "hawk_hawk_payoff": float(self.payoff_entry.get())
            }

            if any(v < 0 for v in params.values()):
                raise ValueError("Negative values not allowed")
            if params["num_food_pairs"] < 1:
                raise ValueError("Food pairs must be ≥1")

            self.status_label.config(text="Running...", foreground="#CBD5E0")
            self.run_button.config(state=tk.DISABLED)

            def thread_target():
                try:
                    result = simulate_hawk_dove(**params)
                    self.result_queue.put(result)
                except Exception as e:
                    self.result_queue.put(e)

            threading.Thread(target=thread_target, daemon=True).start()
            self.check_queue()

        except Exception as e:
            self.status_label.config(text=f"Input Error: {e}", foreground="#FF0000")
            self.run_button.config(state=tk.NORMAL)

    def check_queue(self):
        try:
            result = self.result_queue.get_nowait()
            if isinstance(result, tuple):
                self.plot_results(*result)
                self.status_label.config(text="Simulation Complete", foreground="#48BB78")
            elif isinstance(result, Exception):
                self.status_label.config(text=f"Error: {result}", foreground="#FF0000")
            self.run_button.config(state=tk.NORMAL)
            self.polling_after_id = None
        except queue.Empty:
            self.polling_after_id = self.root.after(100, self.check_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = HawkDoveApp(root)
    root.mainloop()