import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import torch
import pickle
import os
import re
import ast
import numpy as np
import matplotlib
from scipy.stats import gaussian_kde
from DRQI_Laplace2d import PINN
from DRQI_FokkerPlank2d import MPINN
from matplotlib.ticker import MaxNLocator, LogLocator
import math
import matplotlib.font_manager as fm

COLORS = ['#FFA853', '#92B8F9', '#F39EF9', '#7DDE6A']
TYPE01 = ['--', '-.', '-']

# Set global style for journal-quality plots
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22


def parse_log_file(log_path, theoretical=None):
    with open(log_path, 'r') as f:
        content = f.read()

    def extract(pattern, flags=0):
        match = re.search(pattern, content, flags)
        return match.group(1) if match else None

    return {
        'seed': extract(r'Seed: (\d+)'),
        'dimension': extract(r'Model Dimension \(d\): (\d+)'),
        'samples': extract(r'Number of Samples: (\d+)'),
        'optimizer': extract(r'Optimizer: (\w+)'),
        'learning_rate': extract(r'Learning Rate: ([\d\.eE+-]+)'),
        'architecture': extract(r'Model Architecture:\n(.+?)\nNumber', re.DOTALL),
        'algorithm': extract(r'Algorithm: (\w+)'),
        'unique_param': extract(r'Unique Parameters: (.*?)\n'),
        'epochs': extract(r'Max Epochs: (\d+)'),
        'real_epochs': extract(r'Real Epochs: (\d+)'),
        'epsilon1': extract(r'\(eq_loss < epsilon1\): ([\d\.eE+-]+)'),
        'epsilon2': extract(r'\(eq_loss_k - ep_loss_k-1 < epsilon2\): ([\d\.eE+-]+)'),
        'final_loss': extract(r'Loss: ([\d\.eE+-]+)'),
        'eigen_value': extract(r'Eigen Value: ([\d\.eE+-]+)'),
        'lambda_error': abs(float(extract(r'Eigen Value: ([\d\.eE+-]+)'))-theoretical) if theoretical else extract(r'Lambda_Error: ([\d\.eE+-]+)'),
        'eq_loss': extract(r'Equation Loss:([\d\.eE+-]+)'),
        'u_loss': extract(r'U Loss:([\d\.eE+-]+)'),
        'sampling_method': extract(r"Input Sampling Method: (\w+)")
    }


class App:
    def __init__(self, root):
        self.plot_config = {
            "lambda": {"label": "AEE", "color": COLORS[0]},
            "loss": {"label": "LF", "color": COLORS[1]},
            "eq_loss": {"label": "MSR", "color": COLORS[2]},
            "u_loss": {"label": "MSEE", "color": COLORS[3]},
        }
        self.root = root
        self.root.title("PINN Training Visualization")
        self.root.state('zoomed')
        self.mode = tk.StringVar(value="single")
        self.log_files = []
        self.model = None
        self.plot_data = None
        self.theoretical_lambda = None
        self.current_data = None
        self.omega_mode = tk.BooleanVar(value=False)
        self.create_widgets()
        self.problem = "LP"


    def create_widgets(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(top_frame, text="Mode:").pack(side=tk.LEFT)
        tk.Radiobutton(top_frame, text="Single Log", variable=self.mode, value="single", command=self.reset).pack(side=tk.LEFT)
        tk.Radiobutton(top_frame, text="Multi Log", variable=self.mode, value="multi", command=self.reset).pack(side=tk.LEFT)
        tk.Radiobutton(top_frame, text="Comparative Log", variable=self.mode, value="comparative", command=self.reset).pack(
            side=tk.LEFT)
        tk.Button(top_frame, text="Select Log(s)", command=self.load_logs).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Show Details", command=self.show_details).pack(side=tk.LEFT, padx=5)

        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.plot_mode = tk.StringVar(value="combined error")
        self.plot_options = ttk.Combobox(control_frame, textvariable=self.plot_mode,
                                         values=["combined error", "Function (1D only)", "Density"], state="readonly")
        self.plot_options.pack(side=tk.LEFT, padx=10)
        tk.Label(control_frame, text="Sample Size in Density Plot:").pack(side=tk.LEFT, padx=(20, 5))
        self.sample_size_entry = tk.Entry(control_frame, width=6)
        self.sample_size_entry.pack(side=tk.LEFT)
        self.sample_size_entry.insert(0, "5000")
        # Plot checkboxes
        self.plot_lambda = tk.BooleanVar(value=True)
        self.plot_loss = tk.BooleanVar(value=False)
        self.plot_u_loss = tk.BooleanVar(value=False)
        self.plot_eq_loss = tk.BooleanVar(value=False)

        tk.Checkbutton(control_frame, text="λ Error", variable=self.plot_lambda).pack(side=tk.LEFT)
        tk.Checkbutton(control_frame, text="Loss", variable=self.plot_loss).pack(side=tk.LEFT)
        tk.Checkbutton(control_frame, text="U Loss", variable=self.plot_u_loss).pack(side=tk.LEFT)
        tk.Checkbutton(control_frame, text="Eq. Loss", variable=self.plot_eq_loss).pack(side=tk.LEFT)

        tk.Label(control_frame, text="Y Axis Scale:").pack(side=tk.LEFT, padx=(30, 5))
        self.y_scale = tk.StringVar(value="log")
        ttk.Combobox(control_frame, textvariable=self.y_scale, values=["linear", "log"], state="readonly", width=6).pack(side=tk.LEFT)

        tk.Button(control_frame, text="Plot", command=self.plot_selected).pack(side=tk.LEFT, padx=10)


        self.file_list = tk.Listbox(top_frame, height=3, width=60)
        self.file_list.pack(side=tk.LEFT, padx=5)

        tk.Label(top_frame, text="Theoretical λ (use 'pi' for π):").pack(side=tk.LEFT, padx=(20, 0))
        self.lambda_input = tk.Entry(top_frame, width=12)
        self.lambda_input.pack(side=tk.LEFT, padx=5)
        self.lambda_input.insert(0, "1*pi**2")
        tk.Button(top_frame, text="Update Theoretical λ", command=self.update_lambda_theory).pack(side=tk.LEFT)

        tk.Button(top_frame, text="Save Figure", command=self.save_figure).pack(side=tk.LEFT, padx=10)
        tk.Label(top_frame, text="Save Width").pack(side=tk.LEFT)
        self.save_width_entry = tk.Entry(top_frame, width=8)
        self.save_width_entry.pack(side=tk.LEFT)
        self.save_width_entry.insert(0, "7.5")
        tk.Label(top_frame, text="Save Height").pack(side=tk.LEFT)
        self.save_height_entry = tk.Entry(top_frame, width=8)
        self.save_height_entry.pack(side=tk.LEFT)
        self.save_height_entry.insert(0, "5.0")

        # comparative log buttons
        self.comparative_plot_type = tk.StringVar(value="lambda")  # Default: lambda error
        self.comparative_plot_frame = tk.Frame(control_frame)
        self.comparative_plot_frame.pack_forget()  # Default: None

        top_radio_frame = tk.Frame(self.comparative_plot_frame)
        top_radio_frame.pack(side=tk.TOP, anchor='w', pady=2)
        self.radio_lambda = tk.Radiobutton(top_radio_frame, text="λ Error", variable=self.comparative_plot_type,
                                           value="lambda")
        self.radio_loss = tk.Radiobutton(top_radio_frame, text="Loss", variable=self.comparative_plot_type,
                                         value="loss")
        self.radio_u_loss = tk.Radiobutton(top_radio_frame, text="U Loss", variable=self.comparative_plot_type,
                                           value="u_loss")
        self.radio_eq_loss = tk.Radiobutton(top_radio_frame, text="Eq. Loss", variable=self.comparative_plot_type,
                                            value="eq_loss")
        self.radio_u = tk.Radiobutton(top_radio_frame, text="u(x)", variable=self.comparative_plot_type, value="u")
        self.radio_lambda.pack(side=tk.LEFT, padx=5)
        self.radio_loss.pack(side=tk.LEFT, padx=5)
        self.radio_u_loss.pack(side=tk.LEFT, padx=5)
        self.radio_eq_loss.pack(side=tk.LEFT, padx=5)
        self.radio_u.pack(side=tk.LEFT, padx=5)

        omega_radio_frame = tk.Frame(self.comparative_plot_frame)
        omega_radio_frame.pack(side=tk.TOP, anchor='e', pady=2)
        self.radio_omega_lambda = tk.Radiobutton(omega_radio_frame, text="ω-AEE", variable=self.comparative_plot_type,
                                                 value="omega_lambda")
        self.radio_omega_loss = tk.Radiobutton(omega_radio_frame, text="ω-LF", variable=self.comparative_plot_type,
                                               value="omega_loss")
        self.radio_omega_eq_loss = tk.Radiobutton(omega_radio_frame, text="ω-MSR", variable=self.comparative_plot_type,
                                                  value="omega_eq_loss")
        self.radio_omega_u_loss = tk.Radiobutton(omega_radio_frame, text="ω-MSEE", variable=self.comparative_plot_type,
                                                 value="omega_u_loss")
        self.radio_omega_lambda.pack(side=tk.LEFT, padx=5)
        self.radio_omega_loss.pack(side=tk.LEFT, padx=5)
        self.radio_omega_eq_loss.pack(side=tk.LEFT, padx=5)
        self.radio_omega_u_loss.pack(side=tk.LEFT, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)



        # Axis-Control
        self.manual_axis = tk.BooleanVar(value=False)
        tk.Checkbutton(control_frame, text="Manual-Axes Limit", variable=self.manual_axis).pack(side=tk.LEFT, padx=(20, 5))

        self.xmin_entry = tk.Entry(control_frame, width=8)
        self.xmin_entry.pack(side=tk.LEFT)
        self.xmin_entry.insert(0, "")
        tk.Label(control_frame, text="≤ x ≤").pack(side=tk.LEFT)

        self.xmax_entry = tk.Entry(control_frame, width=8)
        self.xmax_entry.pack(side=tk.LEFT)
        self.xmax_entry.insert(0, "")

        self.ymin_entry = tk.Entry(control_frame, width=8)
        self.ymin_entry.pack(side=tk.LEFT, padx=(10, 0))
        self.ymin_entry.insert(0, "")
        tk.Label(control_frame, text="≤ y ≤").pack(side=tk.LEFT)

        self.ymax_entry = tk.Entry(control_frame, width=8)
        self.ymax_entry.pack(side=tk.LEFT)
        self.ymax_entry.insert(0, "")

        self.update_omega_radio_state()

    def apply_manual_axes(self):
        if not self.manual_axis.get():
            return
        try:
            xmin = float(self.xmin_entry.get())
            xmax = float(self.xmax_entry.get())
            ymin = float(self.ymin_entry.get())
            ymax = float(self.ymax_entry.get())
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
        except ValueError:
            messagebox.showerror("Input error", "Check the format of the input value(e.g. 1e-3)")

    def update_lambda_theory(self):
        expr = self.lambda_input.get()
        try:
            expr = expr.replace("pi", f"{np.pi}")
            self.theoretical_lambda = eval(expr)
            messagebox.showinfo("Success", f"Theoretical λ set to {self.theoretical_lambda:.6f}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid expression: {e}")

    def update_omega_radio_state(self):
        state = tk.NORMAL if self.omega_mode.get() else tk.DISABLED
        self.radio_omega_lambda.config(state=state)
        self.radio_omega_loss.config(state=state)
        self.radio_omega_eq_loss.config(state=state)
        self.radio_omega_u_loss.config(state=state)

    def reset(self):
        self.file_list.delete(0, tk.END)
        self.log_files = []
        self.model = None
        self.plot_data = None
        self.ax.clear()
        self.canvas.draw()
        mode = self.mode.get()
        if mode in ["comparative", "multi"]:
            self.comparative_plot_frame.pack(side=tk.LEFT, padx=10)
            self.plot_lambda.set(False)
            self.plot_u_loss.set(False)
            self.plot_loss.set(False)
            self.plot_eq_loss.set(False)
            self.omega_mode.set(mode == "comparative")  # 只有 comparative 模式才允许启用 ω 比较
            self.update_omega_radio_state()
        else:
            self.comparative_plot_frame.pack_forget()
            self.omega_mode.set(False)
            self.update_omega_radio_state()

        if mode == "multi":
            self.manual_axis.set(False)  # 强制取消勾选
            self.xmin_entry.config(state=tk.DISABLED)
            self.xmax_entry.config(state=tk.DISABLED)
            self.ymin_entry.config(state=tk.DISABLED)
            self.ymax_entry.config(state=tk.DISABLED)
        else:
            self.xmin_entry.config(state=tk.NORMAL)
            self.xmax_entry.config(state=tk.NORMAL)
            self.ymin_entry.config(state=tk.NORMAL)
            self.ymax_entry.config(state=tk.NORMAL)

    def load_logs(self):
        mode = self.mode.get()
        if mode == "single":
            path = filedialog.askopenfilename(filetypes=[("Log files", "*.log")])
            if not path:
                return
            self.file_list.delete(0, tk.END)
            self.file_list.insert(tk.END, os.path.basename(path))
            self.log_files = [path]
            self.load_model(path)
        else:
            paths = filedialog.askopenfilenames(filetypes=[("Log files", "*.log")])
            if not paths:
                return
            if self.mode.get() == "comparative" and len(paths) > 4:
                messagebox.showerror("Error", "Comparative mode requires less than 4 logs.")
                return
            self.file_list.delete(0, tk.END)
            for p in paths:
                self.file_list.insert(tk.END, os.path.basename(p))
            self.log_files = list(paths)

    def show_details(self):
        if not self.current_data:
            messagebox.showinfo("Info", "No log data loaded.")
            return
        detail_win = tk.Toplevel(self.root)
        detail_win.title("Log Details")
        detail_win.geometry("500x400")
        text = tk.Text(detail_win, wrap=tk.WORD)
        text.pack(expand=True, fill=tk.BOTH)
        # “key: value”
        for k, v in self.current_data.items():
            text.insert(tk.END, f"{k}: {v}\n")
        text.config(state=tk.DISABLED)

    def load_model(self, log_path):
        data = parse_log_file(log_path, self.theoretical_lambda)
        self.problem = "LP"
        if "_FP_" in log_path:
            self.problem = "FP"
        self.current_data = data  # Save Current Parsed Data
        timestamp = re.search(r"\d{8}_\d{6}", os.path.basename(log_path)).group(0)
        algorithm, d, seed = data["algorithm"], data["dimension"], data["seed"]
        omega = ast.literal_eval(data["unique_param"]).get("omega", None)
        base = f"{algorithm}_{self.problem}_{d}D_OMEGA{omega}_SEED{seed}_Adam_{timestamp}.pkl"
        model_path = os.path.join("models", base)
        if not os.path.exists(model_path):
            messagebox.showerror("Model Missing", f"Model file not found: {model_path}")
            return
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.plot_data = {
            "lambda": np.abs(np.array([x.detach().cpu().item() if torch.is_tensor(x) else float(x) for x in self.model.lambdas]) - self.theoretical_lambda) if self.theoretical_lambda else None,
            "loss": np.array([x.detach().cpu().item() if torch.is_tensor(x) else float(x) for x in self.model.losses]),
            "u_loss": np.array([x.detach().cpu().item() if torch.is_tensor(x) else float(x) for x in self.model.u_losses]),
            "eq_loss": np.array([x.detach().cpu().item() if torch.is_tensor(x) else float(x) for x in self.model.eq_losses]),
        }

    def plot_selected(self):
        yscale = self.y_scale.get()
        # 详细描述某个log的模式
        if self.mode.get() == "single" and self.model:
            mode = self.plot_mode.get()
            if mode == "combined error":
                if self.plot_data is None:
                    messagebox.showwarning("Warning", "Please load model and set theoretical λ first.")
                    return
                self.ax.clear()
                self.ax.grid(True, linestyle='--', color='lightgray')
                x_axis = list(range(len(self.plot_data["loss"])))
                if self.plot_lambda.get() and self.plot_data["lambda"] is not None:
                    self.ax.plot(x_axis, self.plot_data["lambda"], color=self.plot_config["lambda"]["color"], label=self.plot_config["lambda"]["label"])
                if self.plot_loss.get():
                    self.ax.plot(x_axis, self.plot_data["loss"], color=self.plot_config["loss"]["color"], label=self.plot_config["loss"]["label"])
                if self.plot_eq_loss.get():
                    self.ax.plot(x_axis, self.plot_data["eq_loss"], color=self.plot_config["eq_loss"]["color"], label=self.plot_config["eq_loss"]["label"])
                if self.plot_u_loss.get():
                    self.ax.plot(x_axis, self.plot_data["u_loss"], color=self.plot_config["u_loss"]["color"], label=self.plot_config["u_loss"]["label"])
                self.ax.set_title("")
                self.ax.set_xlabel("Epoch")
                self.ax.set_ylabel("Value")
                self.ax.set_yscale(yscale)
                self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

                #self.ax.legend(loc='upper right', framealpha=0.8, fontsize=10)

            elif mode == "Function (1D only)":
                if self.model.d != 1:
                    messagebox.showinfo("Not Supported", "Function plot only supported for 1D models.")
                    return

                try:
                    sample_size = int(self.sample_size_entry.get())
                except:
                    sample_size = 5000

                device = next(self.model.parameters()).device
                d = self.model.d

                amplitude = {"LP": 1, "FP": 2*torch.pi}
                X_raw = torch.rand(sample_size, d, dtype=torch.float64).to(device)*amplitude[self.problem]
                X_list = [X_raw[:, i:i + 1] for i in range(d)]

                with torch.no_grad():
                    u_pred = self.model.net_forward(X_list).cpu().numpy().squeeze()
                    try:
                        u_th = self.model.u_th(X_list).cpu().numpy().squeeze()
                    except Exception as e:
                        print("u_th Calculation Failed", e)
                        u_th = None

                    x_np = X_raw.cpu().numpy().flatten()
                    sort_idx = np.argsort(x_np)
                    x_sorted = x_np[sort_idx]
                    # to keep a positive result
                    u_pred_sorted = np.abs(u_pred[sort_idx])
                    # u_pred needs to be normalized
                    u_pred_sorted = (u_pred_sorted-np.min(u_pred_sorted))/(np.max(u_pred_sorted)-np.min(u_pred_sorted) + 1e-9)
                    # avoid error
                    if u_th is not None:
                        u_th_sorted = u_th[sort_idx]
                        u_th_sorted = np.abs(u_th_sorted)
                        u_th_sorted = (u_th_sorted - np.min(u_th_sorted)) / (np.max(u_th_sorted) - np.min(u_th_sorted) + 1e-9)
                    self.ax.clear()
                    self.ax.grid(True, linestyle='--', color='lightgray')
                    self.ax.plot(x_sorted, u_pred_sorted, label="Prediction", color=COLORS[3])
                    if u_th is not None:
                        self.ax.plot(x_sorted, u_th_sorted, label="Theoretical", color="#000000", linestyle='-.')
                    self.ax.set_xlabel("x")
                    self.ax.set_ylabel("u(x)")
                    self.ax.set_title("Function Plot")
                    self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

                    # self.ax.legend(loc='upper right', framealpha=0.8, fontsize=10)

            elif mode == "Density":
                try:
                    sample_size = int(self.sample_size_entry.get())
                except:
                    sample_size = 5000

                d = self.model.d
                device = next(self.model.parameters()).device

                X_raw = torch.rand(sample_size, d, dtype=torch.float64).to(device)
                X_list = [X_raw[:, i:i + 1] for i in range(d)]

                with torch.no_grad():
                    try:
                        u_pred = self.model.net_forward(X_list).cpu().numpy().squeeze()
                    except Exception as e:
                        print("Prediction Failed", e)
                        return

                    # abs-normalization
                    u_pred = np.abs(u_pred)
                    u_pred = (u_pred - np.min(u_pred)) / (np.max(u_pred) - np.min(u_pred) + 1e-9)

                    try:
                        u_th = self.model.u_th(X_list).cpu().numpy().squeeze()
                        u_th = np.abs(u_th)
                        u_th = (u_th - np.min(u_th)) / (np.max(u_th) - np.min(u_th) + 1e-9)
                    except Exception as e:
                        print("u_th Calculation Failed", e)
                        u_th = None

                    self.ax.clear()
                    self.ax.grid(True, linestyle='--', color='lightgray')
                    pred_density, pred_bins, _ = self.ax.hist(
                        u_pred, bins=100, density=True, alpha=0.7, label="Prediction", color=COLORS[0]
                    )
                    pred_integral = np.sum(pred_density * np.diff(pred_bins))
                    print(f"Integral of predicted density: {pred_integral}")
                    if u_th is not None:
                        th_density, th_bins, _ = self.ax.hist(
                            u_th, bins=100, density=True, alpha=0.3, label="Theoretical", color=COLORS[1]
                        )
                        th_integral = np.sum(th_density * np.diff(th_bins))
                        print(f"Integral of theoretical density: {th_integral}")
                    self.ax.set_xlabel("u")
                    self.ax.set_ylabel("Density")
                    self.ax.set_title("u(x) Density")
                    self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

                    # self.ax.legend(loc='upper right', framealpha=0.8, fontsize=10)

        # 比较模式
        elif self.mode.get() == "comparative":
            self.plot_comparative_log()

        # 点云模式
        elif self.mode.get() == "multi":
            self.plot_point_cloud()

        self.apply_manual_axes()
        self.canvas.draw()

    # compare mode
    def plot_comparative_log(self):
        plot_type = self.comparative_plot_type.get()
        yscale = self.y_scale.get()
        translate = {"lambda": "AEE", "loss": "LF", "eq_loss": "MSR", "u_loss": "MSEE"}
        omega_compare_mode = plot_type.startswith("omega_")
        metric, omega_logs, algorithms = None, None, None
        if omega_compare_mode:
            metric = plot_type.replace("omega_", "")
            omega_logs = {}  # omega: (label, color, data)
        else:
            algorithms = {}

        for i, path in enumerate(self.log_files):
            PROBLEM = "LP"
            if "_FP_" in path:
                PROBLEM = "FP"
            info = parse_log_file(path, self.theoretical_lambda)
            if not info['dimension']:
                continue
            timestamp = re.search(r"\d{8}_\d{6}", os.path.basename(path)).group(0)
            omega = round(float(ast.literal_eval(info["unique_param"]).get("omega", 0.0)), 2)
            algo = info["algorithm"]
            base = f"{algo}_{PROBLEM}_{info['dimension']}D_OMEGA{omega}_SEED{info['seed']}_Adam_{timestamp}.pkl"
            model_path = os.path.join("models", base)

            if not os.path.exists(model_path):
                continue

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            if omega_compare_mode:
                if self.theoretical_lambda:
                    lambda_error = np.abs(np.array(model.lambdas) - self.theoretical_lambda)
                else:
                    lambda_error = [0] * len(model.lambdas)

                metric_data = {
                    "lambda": lambda_error,
                    "loss": model.losses,
                    "eq_loss": [x.detach().cpu().item() for x in model.eq_losses],
                    "u_loss": model.u_losses
                }.get(metric, [])

                if omega not in omega_logs:
                    color_idx = [0.2, 0.4, 0.6, 0.8].index(omega) if omega in [0.2, 0.4, 0.6, 0.8] else i
                    omega_logs[omega] = {
                        "data": metric_data,
                        "color": COLORS[color_idx],
                        "label": f"ω={omega}"
                    }
            else:
                if algo not in algorithms:
                    algorithms[algo] = {"lambda": [], "loss": [], "eq_loss": [], "u_loss": [], "model": None}
                algorithms[algo]["model"] = model
                if self.theoretical_lambda:
                    lambda_error = np.abs(np.array(model.lambdas) - self.theoretical_lambda)
                else:
                    lambda_error = [0] * len(model.lambdas)
                algorithms[algo]["lambda"] = lambda_error
                algorithms[algo]["loss"] = model.losses
                algorithms[algo]["u_loss"] = model.u_losses
                algorithms[algo]["eq_loss"] = [x.detach().cpu().item() for x in model.eq_losses]

        self.ax.clear()
        self.ax.grid(True, linestyle='--', color='lightgray')

        if omega_compare_mode:
            for omega in sorted(omega_logs.keys()):
                entry = omega_logs[omega]
                y_data = entry["data"]
                x_axis = list(range(len(y_data)))
                self.ax.plot(x_axis, y_data, label=entry["label"], color=entry["color"], linestyle='-')
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel(translate.get(metric, metric.upper()))
            self.ax.set_yscale(yscale)
            self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

            # self.ax.legend(loc='upper right', framealpha=0.8, fontsize=10)
        elif plot_type in ["lambda", "loss", "eq_loss", "u_loss"]:
            translate = {"lambda": "AEE", "loss": "LF", "eq_loss": "MSR", "u_loss": "MSEE"}
            for i, algo in enumerate(["DRM", "IPMNN", "DRQI"]):
                if algo not in algorithms:
                    continue
                y_data = algorithms[algo][plot_type]
                x_axis = list(range(len(y_data)))
                self.ax.plot(x_axis, y_data, label=algo, color=COLORS[i], linestyle=TYPE01[i], linewidth="2.5")
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel(translate.get(plot_type, plot_type.upper()))
            self.ax.set_yscale(yscale)
            self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

            # self.ax.legend(loc='upper right', framealpha=0.8, fontsize=10)

        elif plot_type == "u":
            dim = None
            for algo in ["DRM", "IPMNN", "DRQI"]:
                if algo in algorithms and algorithms[algo]["model"]:
                    dim = algorithms[algo]["model"].d
                    break
            if dim != 1:
                messagebox.showinfo("Not Supported", "u(x) comparison only supported for 1D models.")
                return
            for algo in ["DRM", "IPMNN", "DRQI"]:
                if algo not in algorithms or not algorithms[algo]["model"]:
                    continue
                algo_colors = {"DRM": COLORS[0], "IPMNN": COLORS[1], "DRQI": COLORS[2]}
                model = algorithms[algo]["model"]
                x_tensor = model.x[0].detach().cpu()
                u_tensor = model.u.detach().cpu()
                x_np = x_tensor.numpy().flatten()
                u_np = u_tensor.numpy().flatten()
                u_np = np.abs(u_np)
                u_np = (u_np - np.min(u_np)) / (np.max(u_np) - np.min(u_np))
                sort_idx = np.argsort(x_np)
                self.ax.plot(x_np[sort_idx], u_np[sort_idx], label=algo, color=algo_colors[algo])
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("u(x)")
            self.ax.set_title("u(x) Comparison")
            self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            # self.ax.legend(loc='upper right', framealpha=0.8, fontsize=10)

        elif plot_type in ["omega_lambda", "omega_loss", "omega_eq_loss", "omega_u_loss"]:
            omega_map = {0.2: COLORS[0], 0.4: COLORS[1], 0.6: COLORS[2], 0.8: COLORS[3]}
            metric_key = plot_type.replace("omega_", "")
            self.ax.clear()
            self.ax.grid(True, linestyle='--', color='lightgray')
            for path in self.log_files:
                info = parse_log_file(path, self.theoretical_lambda)
                if not info or info["algorithm"] != "DRQI":
                    continue
                omega_val = round(ast.literal_eval(info["unique_param"]).get("omega", None), 2)
                if omega_val not in omega_map:
                    continue
                timestamp = re.search(r"\d{8}_\d{6}", os.path.basename(path)).group(0)
                PROBLEM = "FP" if "_FP_" in path else "LP"
                base = f"DRQI_{PROBLEM}_{info['dimension']}D_OMEGA{omega_val}_SEED{info['seed']}_Adam_{timestamp}.pkl"
                model_path = os.path.join("models", base)
                if not os.path.exists(model_path):
                    continue
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    if metric_key == "lambda":
                        y_data = np.abs(np.array(model.lambdas) - self.theoretical_lambda)
                    elif metric_key == "loss":
                        y_data = model.losses
                    elif metric_key == "eq_loss":
                        y_data = [x.detach().cpu().item() for x in model.eq_losses]
                    elif metric_key == "u_loss":
                        y_data = [x.detach().cpu().item() for x in model.u_losses]
                    else:
                        continue
                    x_data = list(range(len(y_data)))
                    self.ax.plot(x_data, y_data, label=f"ω={omega_val}", color=omega_map[omega_val])
            ylabel_map = {
                "lambda": "AEE",
                "loss": "LF",
                "eq_loss": "MSR",
                "u_loss": "MSEE"
            }
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel(ylabel_map[metric_key])
            self.ax.set_yscale(yscale)
            self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            # self.ax.legend(loc='upper right', framealpha=0.8, fontsize=10)

        self.apply_manual_axes()
        self.canvas.draw()

    def plot_point_cloud(self):
        yscale = self.y_scale.get()
        plot_type = self.comparative_plot_type.get()
        grouped = {"DRM": [], "IPMNN": [], "DRQI": []}

        for path in self.log_files:
            info = parse_log_file(path, self.theoretical_lambda)
            if not info['dimension']:
                continue

            algo = info['algorithm']
            value = None
            if plot_type == "lambda":
                # avoid the mis-record of lambda_error
                value = abs(float(info['eigen_value']) - self.theoretical_lambda)
            elif plot_type == "loss":
                value = float(info['final_loss'])
            elif plot_type == "eq_loss":
                value = float(info['eq_loss'])
            elif plot_type == "u_loss":
                value = float(info['u_loss'])
            else:
                messagebox.showerror("Error", "Multi log mode does not support u(x) point cloud.")
                return
            if value is not None and algo in grouped:
                grouped[algo].append(value)

        self.ax.clear()
        self.ax.grid(True, linestyle='--', color='lightgray')
        for i, (algo, values) in enumerate(grouped.items()):
            if values:
                self.ax.scatter([algo] * len(values), values, label=algo, color=COLORS[i])

        ylabel_map = {
            "lambda": "AEE",
            "loss": "LF",
            "eq_loss": "MSR",
            "u_loss": "MSEE"
        }

        self.ax.set_title(f"{ylabel_map.get(plot_type, 'Metric')} Point Cloud")
        self.ax.set_ylabel(ylabel_map.get(plot_type, 'Value'))
        # self.ax.legend(loc='upper right', framealpha=0.8, fontsize=10)
        self.ax.set_yscale(yscale)
        self.apply_manual_axes()
        self.canvas.draw()

    def save_figure(self):
        default_dir = "./figures"
        os.makedirs(default_dir, exist_ok=True)

        d = self.current_data.get('dimension', 'D') if self.current_data else 'D'
        optimizer = self.current_data.get('optimizer', 'Opt') if self.current_data else 'Opt'
        lr = self.current_data.get('learning_rate', 'LR') if self.current_data else 'LR'
        seed = self.current_data.get('seed', 'Seed') if self.current_data else 'Seed'

        if self.mode.get() == "single":
            y_initial = self.current_data.get("algorithm", "Alg")
        elif self.mode.get() == "comparative":
            translate = {"lambda": "AEE", "loss": "LF",
                         "eq_loss": "MSR", "u_loss": "MSEE"}
            y_initial = translate.get(self.comparative_plot_type.get(), "Y")
        else:
            y_initial = 'Y'

        lr_str = str(lr).replace('.', 'p').replace('e', 'e').replace('+', '').replace('-', 'm')
        default_filename = f"{d}-{optimizer}-LR{lr_str}-SEED{seed}-{y_initial}.png"

        file_path = filedialog.asksaveasfilename(
            initialdir=default_dir,
            initialfile=default_filename,
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            title="Save Figure As"
        )
        if not file_path:
            return

        width = float(self.save_width_entry.get())
        height = float(self.save_height_entry.get())

        # ---------- create clean figure ----------
        fig, ax = plt.subplots(figsize=(width, height), dpi=600)

        # ---------- copy lines ----------
        for line in self.ax.get_lines():
            ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                label=line.get_label()
            )

        # ---------- copy scatter ----------
        for col in self.ax.collections:
            offsets = col.get_offsets()
            if offsets is not None and len(offsets) > 0:
                ax.scatter(
                    offsets[:, 0],
                    offsets[:, 1],
                    s=col.get_sizes(),
                    c=col.get_facecolors(),
                    alpha=col.get_alpha(),
                    label=col.get_label()
                )

        # ---------- copy bars / hist ----------
        for container in self.ax.containers:
            if isinstance(container, matplotlib.container.BarContainer):
                for patch in container.patches:
                    rect = matplotlib.patches.Rectangle(
                        (patch.get_x(), patch.get_y()),
                        patch.get_width(),
                        patch.get_height(),
                        facecolor=patch.get_facecolor(),
                        edgecolor=patch.get_edgecolor(),
                        alpha=patch.get_alpha()
                    )
                    ax.add_patch(rect)

        # ---------- limits & scales ----------
        ax.set_xlim(self.ax.get_xlim())
        ax.set_ylim(self.ax.get_ylim())
        ax.set_yscale(self.ax.get_yscale())

        # ---------- labels (FULL FONT INHERIT) ----------
        xlabel = self.ax.xaxis.label
        ylabel = self.ax.yaxis.label
        title = self.ax.title

        ax.set_xlabel(
            xlabel.get_text(),
            fontsize=xlabel.get_fontsize(),
            fontfamily=xlabel.get_family(),
            fontweight=xlabel.get_weight()
        )
        ax.set_ylabel(
            ylabel.get_text(),
            fontsize=ylabel.get_fontsize(),
            fontfamily=ylabel.get_family(),
            fontweight=ylabel.get_weight()
        )
        ax.set_title(
            title.get_text(),
            fontsize=title.get_fontsize(),
            fontfamily=title.get_family(),
            fontweight=title.get_weight()
        )

        # ---------- ticks ----------
        ax.set_xticks(self.ax.get_xticks())
        ax.set_yticks(self.ax.get_yticks())
        # ---------- y-axis ticks (SAFE for save) ----------
        if ax.get_yscale() == "log":
            # log scale: force ~5 log ticks, never zero
            ax.yaxis.set_major_locator(LogLocator(numticks=6))
        else:
            # linear scale
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.ax.xaxis.get_ticklabels()[0].get_fontsize()
        )
        # ---------- grid ----------
        ax.grid(True, linestyle='--', color='lightgray')

        # ---------- legend ----------
        legend = self.ax.get_legend()
        if legend:
            ax.legend(
                handles=legend.legendHandles,
                labels=[t.get_text() for t in legend.texts],
                fontsize=legend.prop.get_size(),
                framealpha=legend.get_frame().get_alpha()
            )

        fig.tight_layout(pad=1.0)

        try:
            fig.savefig(file_path, dpi=600, bbox_inches='tight')
            messagebox.showinfo("Success", f"Figure saved:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))
        finally:
            plt.close(fig)


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
