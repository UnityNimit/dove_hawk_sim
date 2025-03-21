# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import statistics
from scipy.stats import t, ttest_ind, chi2_contingency
import numpy as np
import time # For progress updates

class SimulationStats:
    def __init__(self):
        self.hawk_populations_over_generations = []
        self.dove_populations_over_generations = []
        self.hawk_survival_rates_per_generation = []
        self.dove_survival_rates_per_generation = []
        self.encounter_counts_total = {'HH': 0, 'HD': 0, 'DD': 0}
        self.food_distribution_all_generations = [] # Stores food values for all agents across all generations
        self.final_hawk_count = 0
        self.final_dove_count = 0

    def record_generation_stats(self, hawks, doves, survival_data, encounters, food_data_this_generation):
        self.hawk_populations_over_generations.append(hawks)
        self.dove_populations_over_generations.append(doves)
        self.hawk_survival_rates_per_generation.append(survival_data['hawk'])
        self.dove_survival_rates_per_generation.append(survival_data['dove'])
        for key in self.encounter_counts_total:
            self.encounter_counts_total[key] += encounters[key]
        self.food_distribution_all_generations.extend(food_data_this_generation)

    def set_final_counts(self, final_hawks, final_doves):
        self.final_hawk_count = final_hawks
        self.final_dove_count = final_doves

    def get_summary_metrics(self):
        def mean_ci(data, confidence=0.95):
            if not data: return (0, 0, 0)
            if len(data) == 1: return (data[0], data[0], data[0]) # Mean is the value, CI is just the value
            
            mean = statistics.mean(data)
            # Ensure there's enough data for stdev
            std_dev_val = statistics.stdev(data) # len(data) >= 2 at this point
            std_err = std_dev_val / (len(data)**0.5)
            
            if std_err == 0 : # Should not happen if len(data) >=2 and values vary, but as a safeguard
                 return (mean, mean, mean)

            dof = len(data) - 1
            t_crit = t.ppf((1 + confidence)/2, dof)
            return (mean, mean - t_crit*std_err, mean + t_crit*std_err)
            
        return {
            'hawk_mean_pop_over_gens': mean_ci(self.hawk_populations_over_generations),
            'dove_mean_pop_over_gens': mean_ci(self.dove_populations_over_generations),
            'avg_hawk_survival_rate': statistics.mean(self.hawk_survival_rates_per_generation) if self.hawk_survival_rates_per_generation else 0,
            'avg_dove_survival_rate': statistics.mean(self.dove_survival_rates_per_generation) if self.dove_survival_rates_per_generation else 0,
            'total_encounters': self.encounter_counts_total,
            'food_stats_overall': (min(self.food_distribution_all_generations), 
                                   statistics.mean(self.food_distribution_all_generations), 
                                   max(self.food_distribution_all_generations)) if self.food_distribution_all_generations else (0,0,0),
            'final_hawk_count': self.final_hawk_count,
            'final_dove_count': self.final_dove_count
        }

def simulate_hawk_dove(
    initial_hawks=50,
    initial_doves=50,
    generations=100,
    num_food_pairs=50,
    hawk_hawk_payoff=0.0, 
    dove_dove_payoff=1.0, 
    hawk_dove_hawk_payoff=2.0, 
    hawk_dove_dove_payoff=0.0, 
    max_population=None,
    run_id=None, 
    progress_queue=None 
):
    stats = SimulationStats()
    population = ["hawk"] * initial_hawks + ["dove"] * initial_doves
    random.shuffle(population)

    hawk_history = []
    dove_history = []
    total_history = []

    for generation_num in range(generations):
        # Send progress update
        if progress_queue and generation_num % 1 == 0: # More frequent updates for debugging
             progress_queue.put({
                 'type': 'progress', 
                 'run_id': run_id, 
                 'generation': generation_num, 
                 'total_generations': generations,
                 'hawk_count': population.count("hawk"), # Add current counts
                 'dove_count': population.count("dove")
            })

        current_pop_size = len(population)
        hawk_count = population.count("hawk")
        dove_count = population.count("dove")

        hawk_history.append(hawk_count)
        dove_history.append(dove_count)
        total_history.append(current_pop_size)

        if current_pop_size == 0: 
            if progress_queue:
                progress_queue.put({'type': 'info', 'run_id': run_id, 'message': f'Population extinct at generation {generation_num}.'})
            remaining_gens = generations - generation_num
            hawk_history.extend([0]*remaining_gens)
            dove_history.extend([0]*remaining_gens)
            total_history.extend([0]*remaining_gens)
            stats.record_generation_stats(0,0,{'hawk':0, 'dove':0}, {'HH':0, 'HD':0, 'DD':0}, []) 
            break

        food_locations = {i: [] for i in range(num_food_pairs)} 
        agent_food_received = {i: 0 for i in range(current_pop_size)} 
        
        for agent_idx in range(current_pop_size):
            chosen_pair_idx = random.randrange(num_food_pairs)
            food_locations[chosen_pair_idx].append(agent_idx)

        current_gen_encounters = {'HH': 0, 'HD': 0, 'DD': 0}
        current_gen_food_values = []

        for pair_idx, agent_indices_at_pair in food_locations.items():
            num_agents_at_pair = len(agent_indices_at_pair)
            
            if num_agents_at_pair == 1: 
                agent_food_received[agent_indices_at_pair[0]] = 2.0 
            elif num_agents_at_pair == 2: 
                idx1, idx2 = agent_indices_at_pair[0], agent_indices_at_pair[1]
                type1, type2 = population[idx1], population[idx2]

                if type1 == "hawk" and type2 == "hawk":
                    agent_food_received[idx1] = hawk_hawk_payoff 
                    agent_food_received[idx2] = hawk_hawk_payoff
                    current_gen_encounters['HH'] += 1
                elif type1 == "dove" and type2 == "dove":
                    agent_food_received[idx1] = dove_dove_payoff 
                    agent_food_received[idx2] = dove_dove_payoff
                    current_gen_encounters['DD'] += 1
                else: 
                    hawk_idx = idx1 if type1 == "hawk" else idx2
                    dove_idx = idx2 if type1 == "hawk" else idx1
                    agent_food_received[hawk_idx] = hawk_dove_hawk_payoff
                    agent_food_received[dove_idx] = hawk_dove_dove_payoff
                    current_gen_encounters['HD'] += 1
            
        for agent_idx in range(current_pop_size):
            current_gen_food_values.append(agent_food_received[agent_idx])

        new_population = []
        survived_hawk_count = 0
        survived_dove_count = 0
        
        for agent_idx in range(current_pop_size):
            food = agent_food_received[agent_idx]
            agent_type = population[agent_idx]
            
            if food >= 1.0: 
                new_population.append(agent_type)
                if agent_type == "hawk": survived_hawk_count +=1
                else: survived_dove_count +=1
                
                if food >= 2.0: 
                    new_population.append(agent_type)
# HACK: Quick fix for edge case, needs a proper solution
            
        current_survival_rates = {
            'hawk': survived_hawk_count / hawk_count if hawk_count > 0 else 0,
            'dove': survived_dove_count / dove_count if dove_count > 0 else 0
        }
        
        stats.record_generation_stats(hawk_count, dove_count, current_survival_rates, current_gen_encounters, current_gen_food_values)
        
        population = new_population
        if max_population and len(population) > max_population:
            population = random.sample(population, max_population)

    final_hawk_count = population.count("hawk")
    final_dove_count = population.count("dove")
    
    if len(hawk_history) <= generations: # Ensure history is full length if sim ended early
         hawk_history.append(final_hawk_count)
         dove_history.append(final_dove_count)
         total_history.append(final_hawk_count + final_dove_count)

    stats.set_final_counts(final_hawk_count, final_dove_count)
    
    if progress_queue: # Final progress message
        progress_queue.put({
            'type': 'progress', 
            'run_id': run_id, 
            'generation': generations, 
            'total_generations': generations,
            'hawk_count': final_hawk_count,
            'dove_count': final_dove_count,
            'message': 'Simulation function complete.'
            })

    return hawk_history, dove_history, total_history, stats

class HawkDoveApp:
    def __init__(self, root_window):
        self.root = root_window
        self.result_queue = queue.Queue()
        self.polling_after_id = None
        
        self.configure_styles()
        self.create_main_widgets()
        self.plot_main_simulation_results([], [], [], None) 
        self.plot_distribution_results(None) 

        self.batch_results_data = [] 
        self.hypothesis_results_A = []
        self.hypothesis_results_B = []


    def configure_styles(self):
        self.root.title("Advanced Hawk-Dove Simulator")
        self.root.geometry("1400x900") 
        self.root.configure(bg="#2D3748") 

        self.style = ttk.Style()
        self.style.theme_use('clam') 
        
        self.style.configure("TFrame", background="#2D3748")
        self.style.configure("TLabel", foreground="#CBD5E0", background="#2D3748", font=('Segoe UI', 10))
        self.style.configure("Header.TLabel", font=('Segoe UI', 14, 'bold'), foreground="#E2E8F0")
        self.style.configure("SubHeader.TLabel", font=('Segoe UI', 12, 'italic'), foreground="#A0AEC0")
        self.style.configure("TEntry", fieldbackground="#E2E8F0", foreground="#1A202C", font=('Segoe UI', 10))
        self.style.configure("TButton", background="#4299E1", foreground="white", font=('Segoe UI', 10, 'bold'), borderwidth=0, padding=5)
        self.style.map("TButton", background=[('active', '#3182CE')])
        self.style.configure("TNotebook", background="#2D3748", borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#4A5568", foreground="#E2E8F0", font=('Segoe UI', 10, 'bold'), padding=[10,5])
        self.style.map("TNotebook.Tab", background=[('selected', '#2D3748')], foreground=[('selected', '#63B3ED')])
        self.style.configure("Treeview", fieldbackground="#4A5568", foreground="#E2E8F0", background="#4A5568")
        self.style.configure("Treeview.Heading", font=('Segoe UI', 10, 'bold'), background="#2D3748", foreground="#CBD5E0")


    def create_main_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(expand=True, fill=tk.BOTH, pady=10)

        self.sim_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.sim_tab, text='Simulation & Analysis')
        self.create_simulation_tab_widgets(self.sim_tab)

        self.dist_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.dist_tab, text='Distribution Visualizer')
        self.create_distribution_tab_widgets(self.dist_tab)

        self.hypo_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.hypo_tab, text='Hypothesis Testing (A/B)')
        self.create_hypothesis_tab_widgets(self.hypo_tab)
        
        self.status_bar_frame = ttk.Frame(main_frame, padding=(5,2))
        self.status_bar_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.global_status_label = ttk.Label(self.status_bar_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.global_status_label.pack(fill=tk.X)


    def create_simulation_tab_widgets(self, parent_tab):
        left_panel = ttk.Frame(parent_tab)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        right_panel = ttk.Frame(parent_tab)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        input_frame = ttk.LabelFrame(left_panel, text="Simulation Parameters", padding="10")
        input_frame.pack(fill=tk.X, pady=5)

        self.sim_params = {}
        param_labels = {
            "initial_hawks": "Initial Hawks:", "initial_doves": "Initial Doves:",
            "generations": "Generations:", "num_food_pairs": "Food Resources (Pairs):",
            "hawk_hawk_payoff": "Hawk-Hawk Food/Agent:", "dove_dove_payoff": "Dove-Dove Food/Agent:",
            "hawk_dove_hawk_payoff": "Hawk (vs Dove) Food:", "hawk_dove_dove_payoff": "Dove (vs Hawk) Food:",
            "max_population": "Max Population (0 for none):"
        }
        default_values = {
            "initial_hawks": "50", "initial_doves": "50", "generations": "100",
            "num_food_pairs": "50", "hawk_hawk_payoff": "0.0", 
            "dove_dove_payoff": "1.0", "hawk_dove_hawk_payoff": "2.0", 
            "hawk_dove_dove_payoff": "0.0", "max_population": "500"
        }

        for i, (key, text) in enumerate(param_labels.items()):
            ttk.Label(input_frame, text=text).grid(row=i, column=0, padx=5, pady=3, sticky="w")
            entry = ttk.Entry(input_frame, width=12)
            entry.insert(0, default_values[key])
            entry.grid(row=i, column=1, padx=5, pady=3, sticky="ew")
            self.sim_params[key] = entry
        
        input_frame.columnconfigure(1, weight=1)

        control_frame = ttk.Frame(left_panel) 
        control_frame.pack(fill=tk.X, pady=10)
        
        self.run_button = ttk.Button(control_frame, text="Run Single Simulation", command=self.run_single_simulation_from_tab)
        self.run_button.pack(fill=tk.X, pady=5)

        self.sim_status_label = ttk.Label(control_frame, text="Status: Ready", anchor="w")
        self.sim_status_label.pack(fill=tk.X, pady=5)
        
        stats_display_frame = ttk.LabelFrame(left_panel, text="Single Run Statistics", padding="10")
        stats_display_frame.pack(fill=tk.X, pady=10)
        self.stats_labels = {}
        metrics_to_display = [
            ('Avg Hawk Pop (over gens)', 'hawk_mean_pop_over_gens'), ('Avg Dove Pop (over gens)', 'dove_mean_pop_over_gens'),
            ('Avg Hawk Survival %', 'avg_hawk_survival_rate'), ('Avg Dove Survival %', 'avg_dove_survival_rate'),
            ('Final Hawk Count', 'final_hawk_count'), ('Final Dove Count', 'final_dove_count'),
            ('Food (Min, Avg, Max)', 'food_stats_overall'),
            ('H-H Encounters', 'total_encounters.HH'), ('H-D Encounters', 'total_encounters.HD'), ('D-D Encounters', 'total_encounters.DD')
        ]
        for i, (label_text, key_path) in enumerate(metrics_to_display):
            ttk.Label(stats_display_frame, text=label_text, anchor='w').grid(row=i, column=0, sticky='w', pady=2)
            val_label = ttk.Label(stats_display_frame, text="N/A", anchor='e', width=20) 
            val_label.grid(row=i, column=1, sticky='ew', pady=2)
            self.stats_labels[key_path] = val_label
        stats_display_frame.columnconfigure(1, weight=1)

        self.fig_sim, self.ax_sim = plt.subplots(figsize=(8, 5), facecolor="#2D3748") 
        self.ax_sim.set_facecolor("#4A5568") 
        self.canvas_sim = FigureCanvasTkAgg(self.fig_sim, master=right_panel)
        self.canvas_sim.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig_sim.tight_layout()

    def create_distribution_tab_widgets(self, parent_tab):
        controls_frame = ttk.Frame(parent_tab, padding="10")
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        plot_frame = ttk.Frame(parent_tab, padding="10")
        plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=5)

        ttk.Label(controls_frame, text="Data to Visualize:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dist_data_var = tk.StringVar(value="Food Distribution (Last Run)")
        dist_options = ["Food Distribution (Last Run)", "Final Hawk Counts (Batch Run)", "Final Dove Counts (Batch Run)"]
        dist_menu = ttk.OptionMenu(controls_frame, self.dist_data_var, self.dist_data_var.get(), *dist_options, command=self.on_dist_data_selection_change)
        dist_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.dist_runs_label = ttk.Label(controls_frame, text="Number of Runs (for batch):")
        self.dist_runs_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dist_runs_entry = ttk.Entry(controls_frame, width=10)
        self.dist_runs_entry.insert(0, "30")
        self.dist_runs_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        self.dist_bins_label = ttk.Label(controls_frame, text="Histogram Bins:")
        self.dist_bins_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.dist_bins_entry = ttk.Entry(controls_frame, width=10)
        self.dist_bins_entry.insert(0, "20")
        self.dist_bins_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        self.run_dist_button = ttk.Button(controls_frame, text="Generate Distribution Plot", command=self.run_distribution_visualization)
        self.run_dist_button.grid(row=1, column=2, columnspan=2, padx=10, pady=10, sticky="ew")
        
        self.dist_status_label = ttk.Label(controls_frame, text="Status: Ready", width=40)
        self.dist_status_label.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky="w")

        controls_frame.columnconfigure(1, weight=1)
        controls_frame.columnconfigure(3, weight=1)
        self.on_dist_data_selection_change() 

        self.fig_dist, self.ax_dist = plt.subplots(figsize=(8, 5), facecolor="#2D3748")
        self.ax_dist.set_facecolor("#4A5568")
        self.canvas_dist = FigureCanvasTkAgg(self.fig_dist, master=plot_frame)
        self.canvas_dist.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig_dist.tight_layout()

    def on_dist_data_selection_change(self, *args):
        selected_option = self.dist_data_var.get()
        if "Batch Run" in selected_option:
            self.dist_runs_entry.config(state=tk.NORMAL)
            self.dist_runs_label.config(state=tk.NORMAL)
        else:
            self.dist_runs_entry.config(state=tk.DISABLED)
            self.dist_runs_label.config(state=tk.DISABLED)

    def create_hypothesis_tab_widgets(self, parent_tab):
        top_frame = ttk.Frame(parent_tab)
        top_frame.pack(fill=tk.X, pady=5)

        self.hypo_params_A = {}
        self.hypo_params_B = {}

        frame_A = ttk.LabelFrame(top_frame, text="Scenario A Parameters", padding="10")
        frame_A.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)
        self.create_scenario_param_inputs(frame_A, self.hypo_params_A, "A")

        frame_B = ttk.LabelFrame(top_frame, text="Scenario B Parameters", padding="10")
        frame_B.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)
        self.create_scenario_param_inputs(frame_B, self.hypo_params_B, "B")
        
        copy_button_frame = ttk.Frame(top_frame) 
        copy_button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(copy_button_frame, text="Copy SimTab Params to A", command=lambda: self.copy_sim_params_to_hypo('A')).pack(side=tk.LEFT, padx=5)
        ttk.Button(copy_button_frame, text="Copy SimTab Params to B", command=lambda: self.copy_sim_params_to_hypo('B')).pack(side=tk.LEFT, padx=5)

        controls_frame = ttk.LabelFrame(parent_tab, text="A/B Test Configuration", padding="10")
        controls_frame.pack(fill=tk.X, pady=10)

        ttk.Label(controls_frame, text="Number of Runs per Scenario:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.hypo_runs_entry = ttk.Entry(controls_frame, width=10)
        self.hypo_runs_entry.insert(0, "30") 
        self.hypo_runs_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(controls_frame, text="Metric to Test:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.hypo_metric_var = tk.StringVar(value="Final Hawk Count")
        metric_options = ["Final Hawk Count", "Final Dove Count", "Total HH Encounters", "Total HD Encounters", "Total DD Encounters"]
        metric_menu = ttk.OptionMenu(controls_frame, self.hypo_metric_var, self.hypo_metric_var.get(), *metric_options)
        metric_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(controls_frame, text="Statistical Test:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.hypo_test_var = tk.StringVar(value="T-test (Independent)")
        test_options = ["T-test (Independent)", "Chi-squared (Encounter Dist.)"] 
        test_menu = ttk.OptionMenu(controls_frame, self.hypo_test_var, self.hypo_test_var.get(), *test_options)
        test_menu.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        
        controls_frame.columnconfigure(1, weight=1)
        controls_frame.columnconfigure(3, weight=1)

        self.run_hypo_button = ttk.Button(controls_frame, text="Run A/B Test & Analyze", command=self.run_hypothesis_test)
        self.run_hypo_button.grid(row=2, column=0, columnspan=4, padx=5, pady=10, sticky="ew")

        self.hypo_status_label = ttk.Label(controls_frame, text="Status: Ready")
        self.hypo_status_label.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky="w")

        results_frame = ttk.LabelFrame(parent_tab, text="A/B Test Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.hypo_results_text = tk.Text(results_frame, height=15, width=80, wrap=tk.WORD, relief=tk.FLAT,
                                         background="#1A202C", foreground="#E2E8F0", font=('Consolas', 10))
        self.hypo_results_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(self.hypo_results_text, command=self.hypo_results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.hypo_results_text.config(yscrollcommand=scrollbar.set)

    def create_scenario_param_inputs(self, parent_frame, param_dict, scenario_prefix):
        param_labels = {
            "initial_hawks": "Initial Hawks:", "initial_doves": "Initial Doves:",
            "generations": "Generations:", "num_food_pairs": "Food Resources:",
            "hawk_hawk_payoff": "H-H Food/Agent:", "dove_dove_payoff": "D-D Food/Agent:",
            "hawk_dove_hawk_payoff": "Hawk (vs Dove) Food:", "hawk_dove_dove_payoff": "Dove (vs Hawk) Food:",
            "max_population": "Max Population:"
        }
        default_values = { 
            "initial_hawks": "50", "initial_doves": "50", "generations": "100",
            "num_food_pairs": "50", "hawk_hawk_payoff": "0.0",
            "dove_dove_payoff": "1.0", "hawk_dove_hawk_payoff": "2.0",
            "hawk_dove_dove_payoff": "0.0", "max_population": "500"
        }
        for i, (key, text) in enumerate(param_labels.items()):
            ttk.Label(parent_frame, text=text).grid(row=i, column=0, padx=5, pady=2, sticky="w")
            entry = ttk.Entry(parent_frame, width=10)
            entry.insert(0, default_values[key])
            entry.grid(row=i, column=1, padx=5, pady=2, sticky="ew")
            param_dict[key] = entry
        parent_frame.columnconfigure(1, weight=1)

    def copy_sim_params_to_hypo(self, scenario_target): 
        target_dict = self.hypo_params_A if scenario_target == 'A' else self.hypo_params_B
        for key, source_entry in self.sim_params.items():
            if key in target_dict:
                target_dict[key].delete(0, tk.END)
                target_dict[key].insert(0, source_entry.get())
        self.global_status_label.config(text=f"Simulation Tab parameters copied to Scenario {scenario_target}.")

    def get_params_from_entries(self, param_entry_dict):
        params = {}
        try:
            for key, entry in param_entry_dict.items():
                val_str = entry.get()
                if key in ["hawk_hawk_payoff", "dove_dove_payoff", "hawk_dove_hawk_payoff", "hawk_dove_dove_payoff"]:
                    params[key] = float(val_str)
                elif key == "max_population":
                    mp = int(val_str)
                    params[key] = mp if mp > 0 else None 
                else:
                    params[key] = int(val_str)
                
                if params[key] is not None and params[key] < 0 and key not in ["hawk_hawk_payoff"]: 
                     raise ValueError(f"{key.replace('_',' ').title()} cannot be negative.")
            if params["num_food_pairs"] < 1:
                raise ValueError("Food resources must be >= 1.")
            if params["generations"] < 1:
                raise ValueError("Generations must be >= 1.")
            return params
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return None

    def run_single_simulation_from_tab(self):
        if self.polling_after_id:
            self.root.after_cancel(self.polling_after_id)
        
        params = self.get_params_from_entries(self.sim_params)
        if params is None:
            self.sim_status_label.config(text="Status: Input Error", foreground="#FF6B6B")
            return

        self.sim_status_label.config(text="Status: Running...", foreground="#CBD5E0")
        self.global_status_label.config(text="Running single simulation...")
        self.run_button.config(state=tk.DISABLED)

        self.batch_results_data = [] 
        self.last_single_run_stats = None 

        # Pass self.result_queue to _execute_simulation_run for all types of runs
        threading.Thread(target=self._execute_simulation_run, args=(params, 'single_sim', 'single_run_01'), daemon=True).start()
        self.check_queue() 

    def _execute_simulation_run(self, params, run_type_tag, run_id=None):
        """ Helper to run simulation and put result or exception in queue. """
        try:
            # Always send a start message
            self.result_queue.put({'type': 'info', 'run_id': run_id, 'message': f'Sim thread started for {run_type_tag}/{run_id}.'})
            
            # Always pass the result_queue for progress updates
            hawk_hist, dove_hist, total_hist, sim_stats_obj = simulate_hawk_dove(**params, run_id=run_id, progress_queue=self.result_queue)
            
            self.result_queue.put({'type': run_type_tag, 'run_id': run_id, 'data': (hawk_hist, dove_hist, total_hist, sim_stats_obj)})
        except Exception as e:
            self.result_queue.put({'type': 'error', 'run_id': run_id, 'error': e, 'context': run_type_tag})
            # Also log to console for dev debugging
            print(f"Exception in simulation thread {run_type_tag}/{run_id}: {e}")
            import traceback
            traceback.print_exc()


    def check_queue(self):
        try:
            msg = self.result_queue.get_nowait()

            if msg['type'] == 'error':
                error_message = f"Error ({msg.get('context', 'operation')}, ID: {msg.get('run_id','N/A')}): {msg['error']}"
                messagebox.showerror("Runtime Error", error_message)
                self.global_status_label.config(text=error_message, foreground="#FF6B6B") # Show error in global status
                
                # Reset relevant button and status label based on context
                if msg.get('context') == 'single_sim':
                    self.sim_status_label.config(text=f"Status: Error", foreground="#FF6B6B")
                    self.run_button.config(state=tk.NORMAL)
                elif msg.get('context') == 'dist_batch' or 'dist_batch_item' in msg.get('context',''):
                    self.dist_status_label.config(text=f"Status: Error during batch.", foreground="#FF6B6B")
                    self.run_dist_button.config(state=tk.NORMAL)
                elif 'hypo_batch' in msg.get('context', '') or 'hypo_A_item' in msg.get('context','') or 'hypo_B_item' in msg.get('context',''):
                    self.hypo_status_label.config(text=f"Status: Error during A/B test.", foreground="#FF6B6B")
                    self.run_hypo_button.config(state=tk.NORMAL)
                # Do not automatically reschedule polling on error; let user re-initiate
                self.polling_after_id = None 
                return 

            elif msg['type'] == 'info': # Handle info messages
                info_text = f"Info ({msg.get('run_id','N/A')}): {msg.get('message','')}"
                self.global_status_label.config(text=info_text)
                # This is just an info message, continue polling for actual results
                self.polling_after_id = self.root.after(100, self.check_queue)
                return # Important: return to not fall through to generic reschedule

            elif msg['type'] == 'progress': # Handle progress messages
                run_id_str = msg.get('run_id', '')
                gen = msg.get('generation',0)
                total_gen = msg.get('total_generations',0)
                h_count = msg.get('hawk_count',0)
                d_count = msg.get('dove_count',0)
                prog_msg_text = msg.get('message','') # Optional extra message from sim
                
                progress_text = f"Run {run_id_str}: Gen {gen}/{total_gen} (H:{h_count} D:{d_count}) {prog_msg_text}"
                
                # Update specific status labels if context matches
                if 'single_sim' in run_id_str or 'single_run' in run_id_str : # Check if run_id indicates single sim
                    self.sim_status_label.config(text=f"Status: {progress_text}")
                elif 'dist_batch' in run_id_str: 
                    self.dist_status_label.config(text=f"Status: {progress_text}")
                elif 'hypo_A' in run_id_str: 
                    self.hypo_status_label.config(text=f"Status: Scenario A - {progress_text}")
                elif 'hypo_B' in run_id_str: 
                    self.hypo_status_label.config(text=f"Status: Scenario B - {progress_text}")
                
                self.global_status_label.config(text=f"Progress: {progress_text}")
                self.polling_after_id = self.root.after(100, self.check_queue) # Continue polling
                return # Important: return

            elif msg['type'] == 'single_sim':
                hawk_hist, dove_hist, total_hist, sim_stats_obj = msg['data']
                self.last_single_run_stats = sim_stats_obj 
                self.plot_main_simulation_results(hawk_hist, dove_hist, total_hist, sim_stats_obj)
                self.sim_status_label.config(text="Status: Simulation Complete", foreground="#81E6D9")
                self.run_button.config(state=tk.NORMAL)
                self.global_status_label.config(text="Single simulation complete.")
            
            elif msg['type'] == 'dist_batch_run_completed': 
                sim_stats_obj = msg['data'][3] 
                self.batch_results_data.append(sim_stats_obj) 
            
            elif msg['type'] == 'dist_batch_finished': 
                self.plot_distribution_results(self.batch_results_data)
                self.dist_status_label.config(text=f"Status: Batch complete ({len(self.batch_results_data)} runs). Plot generated.", foreground="#81E6D9")
                self.run_dist_button.config(state=tk.NORMAL)
                self.global_status_label.config(text="Distribution batch run finished.")

            elif msg['type'] == 'hypo_batch_A_run_completed':
                sim_stats_obj = msg['data'][3]
                self.hypothesis_results_A.append(sim_stats_obj)
            
            elif msg['type'] == 'hypo_batch_B_run_completed':
                sim_stats_obj = msg['data'][3]
                self.hypothesis_results_B.append(sim_stats_obj)

            elif msg['type'] == 'hypo_batch_finished': 
                self.perform_and_display_statistical_test()
                self.hypo_status_label.config(text=f"Status: A/B Test Complete. Results below.", foreground="#81E6D9")
                self.run_hypo_button.config(state=tk.NORMAL)
                self.global_status_label.config(text="A/B Hypothesis test finished.")
            
            # If a message was processed and it wasn't an error/info/progress that returned early, reschedule polling.
            self.polling_after_id = self.root.after(100, self.check_queue)

# TODO: Refactor this section for clarity
        except queue.Empty:
            still_running = self.run_button['state'] == tk.DISABLED or \
                            self.run_dist_button['state'] == tk.DISABLED or \
                            self.run_hypo_button['state'] == tk.DISABLED
            if still_running:
                 self.polling_after_id = self.root.after(100, self.check_queue)
            else: 
                 self.polling_after_id = None
        except Exception as e: 
            messagebox.showerror("Queue Processing Error", f"An error occurred in check_queue: {e}")
            print(f"Exception in check_queue: {e}") # For dev debugging
            import traceback
            traceback.print_exc()
            # Reset UI states to be safe
            self.run_button.config(state=tk.NORMAL)
            self.run_dist_button.config(state=tk.NORMAL)
            self.run_hypo_button.config(state=tk.NORMAL)
            self.sim_status_label.config(text="Status: Error", foreground="#FF6B6B")
            self.dist_status_label.config(text="Status: Error", foreground="#FF6B6B")
            self.hypo_status_label.config(text="Status: Error", foreground="#FF6B6B")
            self.global_status_label.config(text="Critical error in UI update.")
            self.polling_after_id = None # Stop polling


    def plot_main_simulation_results(self, hawks_hist, doves_hist, total_hist, sim_stats_obj):
        self.ax_sim.clear()
        if hawks_hist: 
            generations_axis = range(len(hawks_hist))
            self.ax_sim.plot(generations_axis, hawks_hist, label="Hawks", color="#F56565") 
            self.ax_sim.plot(generations_axis, doves_hist, label="Doves", color="#48BB78") 
            self.ax_sim.plot(generations_axis, total_hist, label="Total", color="#CBD5E0", linestyle=':')
            if any(h > 0 for h in total_hist) or any(d > 0 for d in total_hist) : # only set bottom if there is data
                 self.ax_sim.set_ylim(bottom=0)
            else: # if all values are zero, set some default y-limit to avoid issues
                 self.ax_sim.set_ylim(0, 1)

        
        self.ax_sim.set_xlabel("Generation", color="#CBD5E0")
        self.ax_sim.set_ylabel("Population", color="#CBD5E0")
        self.ax_sim.set_title("Population Dynamics", color="#F7FAFC", fontsize=14)
        
        self.ax_sim.tick_params(axis='x', colors='#CBD5E0')
        self.ax_sim.tick_params(axis='y', colors='#CBD5E0')
        self.ax_sim.spines['top'].set_color('#2D3748') 
        self.ax_sim.spines['right'].set_color('#2D3748')
        self.ax_sim.spines['left'].set_color('#CBD5E0')
        self.ax_sim.spines['bottom'].set_color('#CBD5E0')

        if hawks_hist: 
            self.ax_sim.legend(facecolor='#4A5568', edgecolor='#CBD5E0', labelcolor="#F7FAFC")
        
        self.fig_sim.tight_layout()
        self.canvas_sim.draw()

        if sim_stats_obj:
            metrics_data = sim_stats_obj.get_summary_metrics()
            for key_path, label_widget in self.stats_labels.items():
                try:
                    value = metrics_data
                    for part in key_path.split('.'): 
                        value = value[part]
                    
                    if isinstance(value, tuple) and len(value) == 3 and isinstance(value[0], (int, float)): 
                        text_val = f"{value[1]:.2f} (CI: {value[0]:.2f}-{value[2]:.2f})" if key_path.endswith("_mean_pop_over_gens") else f"{value[0]:.1f}, {value[1]:.1f}, {value[2]:.1f}"
                    elif isinstance(value, float) and ('survival_rate' in key_path):
                        text_val = f"{value:.2%}"
                    elif isinstance(value, (int, float)):
                        text_val = f"{value:.0f}" if isinstance(value, int) else f"{value:.2f}"
                    else:
                        text_val = str(value)
                    label_widget.config(text=text_val)
                except (KeyError, TypeError, IndexError):
                    label_widget.config(text="N/A")
        else: 
             for label_widget in self.stats_labels.values():
                label_widget.config(text="N/A")

    def run_distribution_visualization(self):
        selected_option = self.dist_data_var.get()
        
        try:
            bins = int(self.dist_bins_entry.get())
            if bins <= 0: raise ValueError("Bins must be positive.")
        except ValueError:
            messagebox.showerror("Input Error", "Number of bins must be a positive integer.")
            return

        if "Batch Run" in selected_option:
            try:
                num_runs = int(self.dist_runs_entry.get())
                if num_runs <= 0: raise ValueError("Number of runs must be positive.")
            except ValueError:
                messagebox.showerror("Input Error", "Number of runs for batch must be a positive integer.")
                return

            sim_params = self.get_params_from_entries(self.sim_params) 
            if sim_params is None:
                self.dist_status_label.config(text="Status: Sim params input error.", foreground="#FF6B6B")
                return

            self.dist_status_label.config(text=f"Status: Starting batch of {num_runs} runs...", foreground="#CBD5E0")
            self.global_status_label.config(text=f"Running batch for distribution ({num_runs} runs)...")
            self.run_dist_button.config(state=tk.DISABLED)
            self.batch_results_data.clear() 

            def batch_runner_thread():
                for i in range(num_runs):
                    run_id = f"dist_batch_{i+1}"
                    try:
                        # Pass self.result_queue for progress updates
                        hawk_h, dove_h, total_h, s_stats = simulate_hawk_dove(**sim_params.copy(), run_id=run_id, progress_queue=self.result_queue)
                        self.result_queue.put({'type': 'dist_batch_run_completed', 'run_id': run_id, 'data': (hawk_h, dove_h, total_h, s_stats)})
                    except Exception as e_run:
                        self.result_queue.put({'type': 'error', 'run_id': run_id, 'error': e_run, 'context': 'dist_batch_item'})
                        break 
                self.result_queue.put({'type': 'dist_batch_finished'}) 

            threading.Thread(target=batch_runner_thread, daemon=True).start()
            self.check_queue() 

        else: 
            if hasattr(self, 'last_single_run_stats') and self.last_single_run_stats:
                self.plot_distribution_results(self.last_single_run_stats) 
                self.dist_status_label.config(text="Status: Plot generated for last single run.", foreground="#81E6D9")
            else:
                messagebox.showinfo("No Data", "Please run a single simulation first to generate data for food distribution.")
                self.dist_status_label.config(text="Status: No data from single run.", foreground="#FFCC00")


    def plot_distribution_results(self, results_data, bins_override=None): 
        self.ax_dist.clear()
        selected_option = self.dist_data_var.get()
        title = selected_option
        data_to_plot = []
        
        try:
            bins = int(self.dist_bins_entry.get()) if bins_override is None else bins_override
        except ValueError:
            bins = 20 

        if not results_data:
            self.ax_dist.text(0.5, 0.5, "No data to display.\nRun simulation or batch.", 
                              ha='center', va='center', color="#CBD5E0", fontsize=12)
        elif selected_option == "Food Distribution (Last Run)":
            if isinstance(results_data, SimulationStats): 
                data_to_plot = results_data.food_distribution_all_generations
                title = "Food Distribution (Last Single Run)"
            else: 
                 self.ax_dist.text(0.5, 0.5, "Run a single simulation first.", ha='center', va='center', color="#CBD5E0")
        
        elif selected_option == "Final Hawk Counts (Batch Run)":
            if isinstance(results_data, list): 
                data_to_plot = [s.final_hawk_count for s in results_data if isinstance(s, SimulationStats)]
                title = f"Distribution of Final Hawk Counts ({len(data_to_plot)} runs)"
        
        elif selected_option == "Final Dove Counts (Batch Run)":
            if isinstance(results_data, list): 
                data_to_plot = [s.final_dove_count for s in results_data if isinstance(s, SimulationStats)]
                title = f"Distribution of Final Dove Counts ({len(data_to_plot)} runs)"

        if data_to_plot:
            self.ax_dist.hist(data_to_plot, bins=bins, color="#63B3ED", edgecolor="#2D3748", alpha=0.8)
            if len(data_to_plot) > 1: # Mean/median only make sense for multiple data points
                mean_val = np.mean(data_to_plot)
                median_val = np.median(data_to_plot)
                self.ax_dist.axvline(mean_val, color='#F6E05E', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
                self.ax_dist.axvline(median_val, color='#A0AEC0', linestyle='dotted', linewidth=2, label=f'Median: {median_val:.2f}')
                self.ax_dist.legend(facecolor='#4A5568', edgecolor='#CBD5E0', labelcolor="#F7FAFC")
        else: # If data_to_plot ended up empty after filtering
             self.ax_dist.text(0.5, 0.5, "Not enough data for this distribution.", ha='center', va='center', color="#CBD5E0")


        self.ax_dist.set_title(title, color="#F7FAFC", fontsize=14)
        self.ax_dist.set_xlabel("Value", color="#CBD5E0")
        self.ax_dist.set_ylabel("Frequency", color="#CBD5E0")
        self.ax_dist.tick_params(axis='x', colors='#CBD5E0')
        self.ax_dist.tick_params(axis='y', colors='#CBD5E0')
        self.ax_dist.grid(True, linestyle='--', alpha=0.3, color="#A0AEC0")
        self.fig_dist.tight_layout()
        self.canvas_dist.draw()

    def run_hypothesis_test(self):
        try:
            num_runs_per_scenario = int(self.hypo_runs_entry.get())
            if num_runs_per_scenario <= 1: 
                messagebox.showerror("Input Error", "Number of runs per scenario must be at least 2 for meaningful tests (preferably >10).")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Number of runs must be an integer.")
            return

        params_A = self.get_params_from_entries(self.hypo_params_A)
        params_B = self.get_params_from_entries(self.hypo_params_B)

        if params_A is None or params_B is None:
            self.hypo_status_label.config(text="Status: Parameter input error for A or B.", foreground="#FF6B6B")
            return

        self.hypo_status_label.config(text=f"Status: Starting A/B test ({num_runs_per_scenario} runs/scenario)...", foreground="#CBD5E0")
        self.global_status_label.config(text=f"Running A/B Hypothesis Test...")
        self.run_hypo_button.config(state=tk.DISABLED)
        self.hypothesis_results_A.clear()
        self.hypothesis_results_B.clear()
        self.hypo_results_text.delete('1.0', tk.END) 

        def ab_test_runner_thread():
            for i in range(num_runs_per_scenario):
                run_id = f"hypo_A_{i+1}"
                try:
                    h_h, d_h, t_h, s_stats = simulate_hawk_dove(**params_A.copy(), run_id=run_id, progress_queue=self.result_queue)
                    self.result_queue.put({'type': 'hypo_batch_A_run_completed', 'run_id': run_id, 'data': (h_h,d_h,t_h,s_stats)})
                except Exception as e_run_A:
                    self.result_queue.put({'type': 'error', 'run_id': run_id, 'error': e_run_A, 'context': 'hypo_A_item'})
                    return 
            
            for i in range(num_runs_per_scenario):
                run_id = f"hypo_B_{i+1}"
                try:
                    h_h, d_h, t_h, s_stats = simulate_hawk_dove(**params_B.copy(), run_id=run_id, progress_queue=self.result_queue)
                    self.result_queue.put({'type': 'hypo_batch_B_run_completed', 'run_id': run_id, 'data': (h_h,d_h,t_h,s_stats)})
                except Exception as e_run_B:
                    self.result_queue.put({'type': 'error', 'run_id': run_id, 'error': e_run_B, 'context': 'hypo_B_item'})
                    return 
            
            self.result_queue.put({'type': 'hypo_batch_finished'}) 

        threading.Thread(target=ab_test_runner_thread, daemon=True).start()
        self.check_queue()


    def perform_and_display_statistical_test(self):
        self.hypo_results_text.delete('1.0', tk.END)
        
        if not self.hypothesis_results_A or not self.hypothesis_results_B:
            self.hypo_results_text.insert(tk.END, "Error: Not enough data collected for scenarios A or B.\n")
            return

        num_runs_A = len(self.hypothesis_results_A)
        num_runs_B = len(self.hypothesis_results_B)
        self.hypo_results_text.insert(tk.END, f"--- A/B Test Results ---\n")
        self.hypo_results_text.insert(tk.END, f"Scenario A: {num_runs_A} runs completed.\n")
        self.hypo_results_text.insert(tk.END, f"Scenario B: {num_runs_B} runs completed.\n\n")

        selected_metric = self.hypo_metric_var.get()
        selected_test = self.hypo_test_var.get()

        data_A, data_B = [], []
        metric_label = selected_metric

        if selected_metric == "Final Hawk Count":
            data_A = [s.final_hawk_count for s in self.hypothesis_results_A]
            data_B = [s.final_hawk_count for s in self.hypothesis_results_B]
        elif selected_metric == "Final Dove Count":
            data_A = [s.final_dove_count for s in self.hypothesis_results_A]
            data_B = [s.final_dove_count for s in self.hypothesis_results_B]
        elif "Encounters" in selected_metric: 
            pass 
        else:
            self.hypo_results_text.insert(tk.END, f"Metric '{selected_metric}' not yet implemented for testing.\n")
            return

        if data_A and data_B: 
            self.hypo_results_text.insert(tk.END, f"Metric: {metric_label}\n")
            self.hypo_results_text.insert(tk.END, f"  Scenario A - Mean: {np.mean(data_A):.2f}, StdDev: {np.std(data_A):.2f}, Median: {np.median(data_A):.2f}\n")
            self.hypo_results_text.insert(tk.END, f"  Scenario B - Mean: {np.mean(data_B):.2f}, StdDev: {np.std(data_B):.2f}, Median: {np.median(data_B):.2f}\n\n")


        if selected_test == "T-test (Independent)":
            if not data_A or not data_B:
                self.hypo_results_text.insert(tk.END, "Not enough data for T-test on the selected metric.\n")
                return
            if len(data_A) < 2 or len(data_B) < 2:
                 self.hypo_results_text.insert(tk.END, "Not enough data points (need at least 2 per scenario) for T-test.\n")
                 return

            t_stat, p_value = ttest_ind(data_A, data_B, equal_var=False) 
            self.hypo_results_text.insert(tk.END, f"--- {selected_test} for {metric_label} ---\n")
            self.hypo_results_text.insert(tk.END, f"  T-statistic: {t_stat:.4f}\n")
            self.hypo_results_text.insert(tk.END, f"  P-value: {p_value:.4f}\n")
            alpha = 0.05
            if p_value < alpha:
                self.hypo_results_text.insert(tk.END, f"  Conclusion: Significant difference detected between Scenario A and B (p < {alpha}).\n")
            else:
                self.hypo_results_text.insert(tk.END, f"  Conclusion: No significant difference detected (p >= {alpha}).\n")

        elif selected_test == "Chi-squared (Encounter Dist.)":
            enc_A = {'HH': 0, 'HD': 0, 'DD': 0}
            enc_B = {'HH': 0, 'HD': 0, 'DD': 0}
            for s in self.hypothesis_results_A:
                for k_enc in enc_A: enc_A[k_enc] += s.encounter_counts_total[k_enc]
            for s in self.hypothesis_results_B:
                for k_enc in enc_B: enc_B[k_enc] += s.encounter_counts_total[k_enc]
            
            observed_table = [
                [enc_A['HH'], enc_A['HD'], enc_A['DD']],
                [enc_B['HH'], enc_B['HD'], enc_B['DD']]
            ]
            
            self.hypo_results_text.insert(tk.END, f"--- {selected_test} for Encounter Type Distributions ---\n")
            self.hypo_results_text.insert(tk.END, f"  Observed Frequencies:\n")
            self.hypo_results_text.insert(tk.END, f"    Scenario A: HH={enc_A['HH']}, HD={enc_A['HD']}, DD={enc_A['DD']}\n")
            self.hypo_results_text.insert(tk.END, f"    Scenario B: HH={enc_B['HH']}, HD={enc_B['HD']}, DD={enc_B['DD']}\n\n")

            if any(sum(row) == 0 for row in observed_table) or \
               (all(enc_A[k_enc]==0 for k_enc in enc_A) and all(enc_B[k_enc]==0 for k_enc in enc_B)) : 
                self.hypo_results_text.insert(tk.END, "  Chi-squared test cannot be performed: one or more scenarios have zero total encounters, or all categories are zero across both.\n")
            else:
                try:
                    chi2_stat, p_value, dof, expected_freq = chi2_contingency(observed_table)
                    self.hypo_results_text.insert(tk.END, f"  Chi-squared Statistic: {chi2_stat:.4f}\n")
                    self.hypo_results_text.insert(tk.END, f"  P-value: {p_value:.4f}\n")
                    self.hypo_results_text.insert(tk.END, f"  Degrees of Freedom: {dof}\n")
                    alpha = 0.05
                    if p_value < alpha:
                        self.hypo_results_text.insert(tk.END, f"  Conclusion: Significant difference in encounter distributions between Scenarios (p < {alpha}).\n")
                    else:
                        self.hypo_results_text.insert(tk.END, f"  Conclusion: No significant difference in encounter distributions (p >= {alpha}).\n")
                except ValueError as e_chi: 
                     self.hypo_results_text.insert(tk.END, f"  Chi-squared test could not be performed: {e_chi}\n")
        else:
            self.hypo_results_text.insert(tk.END, f"Test '{selected_test}' not implemented yet.\n")

        self.hypo_results_text.insert(tk.END, "\n--- End of Report ---\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = HawkDoveApp(root)
    root.mainloop()

# Simulated change for commit history

# Simulated change for commit history
