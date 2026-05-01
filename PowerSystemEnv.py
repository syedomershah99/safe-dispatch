# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:22:27 2026

@author: morufdee
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from IPython.display import clear_output, display
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib.animation as animation
from io import BytesIO
import PIL.Image as Image
import imageio
import time
import networkx as nx

def generate_system_network():
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1.5, 5.5)
    ax.axis('off')

    # Bus Layout: 1: Birch, 2: Elm, 3: Pine, 4: Maple
    bus_coords = {1: (0, 4), 2: (5, 4), 3: (0, 0), 4: (5, 0)}
    bus_names = {1: "Birch", 2: "Elm", 3: "Pine", 4: "Maple"}
    
    # Line Data: (From, To, R, X, B_total/2)
    lines = [
        (1, 2, 0.01008, 0.05040, 0.05125),
        (1, 3, 0.00744, 0.03720, 0.03875),
        (2, 4, 0.00744, 0.03720, 0.03875),
        (3, 4, 0.01272, 0.06360, 0.06375)
    ]

    # Equipment Data
    loads = {1: 50, 2: 170, 3: 200, 4: 80} # MW
    # Modification: Bus 2 has G2 and G3
    gens = {1: ["G1 (Slack)"], 2: ["G2 (Cheap)", "G3 (Exp)"], 4: ["G4"]}

    # Draw Transmission Lines (Branches)
    for f, t, r, x, b in lines:
        start, end = bus_coords[f], bus_coords[t]
        ax.plot([start[0], end[0]], [start[1], end[1]], color='#2c3e50', linewidth=2.5, zorder=1)
        
        # Labeling Line Parameters (R, X, B/2)
        mx, my = (start[0]+end[0])/2, (start[1]+end[1])/2
        param_text = f"z = {r} + j{x}\ny/2 = j{b}"
        
        offset = 0.25 if start[0] == end[0] else -0.3
        ax.text(mx + (0.3 if start[0]==end[0] else 0), my + (0 if start[0]==end[0] else offset), 
                param_text, fontsize=8, ha='center', fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Draw Buses and Equipment
    for i, (x, y) in bus_coords.items():
        # Draw Bus Bar (Matching Render style)
        ax.add_patch(Rectangle((x-0.5, y-0.05), 1.0, 0.1, color='#2c3e50', zorder=5))
        ax.text(x, y+0.2, f"BUS {i}: {bus_names[i]}", fontsize=11, fontweight='black', ha='center')

        # Draw Generators (Circles)
        if i in gens:
            for idx, g_name in enumerate(gens[i]):
                x_off = (idx - 0.5) * 0.7 if len(gens[i]) > 1 else 0
                y_gen = y + 0.6 if y >= 4 else y + 0.6 # Position above
                
                ax.add_patch(Circle((x + x_off, y_gen), 0.2, color='#27ae60', alpha=0.9, zorder=6))
                ax.text(x + x_off, y_gen + 0.3, g_name, fontsize=8, ha='center', fontweight='bold', color='#27ae60')
                # Connection line
                ax.plot([x + x_off, x + x_off], [y, y_gen-0.2], color='#27ae60', linewidth=1.5)

        # Draw Loads (Triangles)
        if i in loads:
            y_load = y - 0.6
            tri_pts = np.array([[x, y_load], [x-0.2, y_load+0.3], [x+0.2, y_load+0.3]])
            ax.add_patch(Polygon(tri_pts, color='#2980b9', alpha=0.9, zorder=6))
            ax.text(x, y_load - 0.3, f"L: {loads[i]} MW", fontsize=9, ha='center', fontweight='bold', color='#2980b9')
            # Connection line
            ax.plot([x, x], [y, y_load+0.3], color='#2980b9', linewidth=1.5)

    # Add Legend and Title
    # plt.title(f"Modified 4-Bus System Configuration (case4gs)\n(Nodal parameters based on Grainger & Stevenson)", 
    #          fontsize=14, fontweight='black', pad=30)
    
    # Informative Footer
    ax.text(2.5, -1.2, "System Base: 100 MVA, 230 kV | Impedances in p.u.", 
            ha='center', fontsize=10, fontstyle='italic', bbox=dict(facecolor='#f1c40f', alpha=0.2))

    # Save Files
    plt.savefig("4-bus_system_network.png", dpi=300, bbox_inches='tight')
    plt.savefig("4-bus_system_network.pdf", bbox_inches='tight')
    print("System Network saved: .png and .pdf")
    plt.show()

def build_matrices(n_buses, lines, slack_bus=0):
    """
    lines: list of tuples (from_bus, to_bus, reactance_x)
    Returns B_bus and PTDF matrix.
    """
    # Build Adjacency/Incidence Matrix A and Diagonal Susceptance Matrix B_line
    n_lines = len(lines)
    A = np.zeros((n_lines, n_buses))
    B_line = np.zeros((n_lines, n_lines))
    
    for idx, (f, t, x) in enumerate(lines):
        A[idx, f] = 1
        A[idx, t] = -1
        B_line[idx, idx] = 1/x
        
    # Build B_bus = A.T @ B_line @ A
    B_bus = A.T @ B_line @ A
    
    # 3. Calculate PTDF = B_line @ A @ inv(B_bus_reduced)
    # We remove the slack bus row/column to make it non-singular
    non_slack = [i for i in range(n_buses) if i != slack_bus]
    B_bus_red = B_bus[np.ix_(non_slack, non_slack)]
    B_bus_inv_red = np.linalg.inv(B_bus_red)
    
    # Expand back to full size with zeros for slack bus
    B_bus_inv = np.zeros((n_buses, n_buses))
    B_bus_inv[np.ix_(non_slack, non_slack)] = B_bus_inv_red
    
    PTDF = B_line @ A @ B_bus_inv
    
    return B_bus, PTDF, A

def vec(x, n=None):
    x = np.asarray(x).reshape(-1)
    if n is not None:
        assert x.size == n, f"Expected size {n}, got {x.size}"
    return x

class PowerSystemEnv(gym.Env):
    
    # ==============================================================================
    # EXPERIMENTAL CONFIGURATION NOTE:
    # ------------------------------------------------------------------------------
    # FOR BENCHMARKING (Comparison with OSQP Solver):
    #   - Set self.load_drift = 0.05 (5% drift)
    #   - Set self.F_max[0] = 1.2 (Line 1-3 relaxation)
    #   - Set self.current_Pd clipping to [0.95, 1.05]
    #
    # FOR STRESS TESTING / TRAINING (Original Agent Environment):
    #   - Set self.load_drift = 0.20 (20% drift)
    #   - Set self.F_max[0] = 1.0 (Original thermal limit)
    #   - Set self.current_Pd clipping to [0.80, 1.20]
    # ==============================================================================

    def __init__(self):
        super(PowerSystemEnv, self).__init__()
        self.np_random = np.random.default_rng()
        # --- System Constants ---
        self.n_buses = 4
        self.n_gens = 4
        self.lines_data = [(0, 1, 0.0504), (0, 2, 0.0372), (1, 3, 0.0372), (2, 3, 0.0636)]
        self.F_max = np.array([1.2, 1.0, 0.8, 1.5]) # Targeted Bottleneck on Line 2-4
        
        # Mapping Generators to Buses (G1->Bus0, G2->Bus1, G3->Bus1, G4->Bus3)
        self.gen_bus_map = [0, 1, 1, 3]
        self.P_min = np.array([0.0, 0.0, 0.0, 0.0])
        self.P_max = np.array([2.0, 1.5, 1.0, 1.5])
        self.P_d_base = np.array([0.5, 1.7, 2.0, 0.8])
        
        # Pre-calculate Physics (Universal Solver)
        self.B_bus, self.PTDF, self.A = build_matrices(self.n_buses, self.lines_data)
        
        # Pre-calculate Mapping Matrix M (N_bus x N_gen)
        self.M = np.zeros((self.n_buses, self.n_gens))
        for i, bus_idx in enumerate(self.gen_bus_map):
            self.M[bus_idx, i] = 1.0

        # --- Spaces ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_gens,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(12,), dtype=np.float32)

        # --- Episode Lifecycle ---
        self.max_steps = 100       # The "Clock" - length of one tracking scenario
        self.current_step = 0
        self.load_drift = 0.02    # 2% max random change in load per step

        # State initialization
        self.current_Pg = None
        self.current_Pd = None
        # --- Placeholder for animation frames ---
        self.frames = []
        
    def solve_traditional_sced(self, Pd):
        """Standard Quadratic Programming (QP) solver for SCED benchmark."""
        Pg = cp.Variable(self.n_gens)
        
        # Generator cost coefficients
        a = np.array([0.1085, 0.1085, 0.1085, 0.1085])
        b = np.array([0.0832, 0.0260, 0.0832, 0.0260])
        c = np.array([0.008, 0.006, 0.008, 0.006])
        cost_multiplier = 5000.0
        
        objective = cp.Minimize(cp.sum(cp.multiply(a, cp.square(Pg)) + cp.multiply(b, Pg) + c) * cost_multiplier)
        
        constraints = [
            cp.sum(Pg) == cp.sum(Pd),
            Pg >= self.P_min,
            Pg <= self.P_max,
            self.PTDF @ (self.M @ Pg - Pd) <= self.F_max,
            self.PTDF @ (self.M @ Pg - Pd) >= -self.F_max
        ]
        
        prob = cp.Problem(objective, constraints)
        
        prob.solve(solver=cp.OSQP, verbose=False)
        
        if Pg.value is None:
            return None, None
            
        return Pg.value, prob.value
    

    def reset(self, seed=None, options=None):
        # Handle seeding
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            
        # Reset the Episode Clock
        self.current_step = 0
            
        # Randomize Load +/- 10% around base
        self.current_Pd = (self.P_d_base * self.np_random.uniform(0.9, 1.1, size=self.n_buses)).flatten()
        
        # Initial Pg: Midpoint projected to FEASIBLE space
        # This ensures the first frame of the render is already "Safe"
        midpoint_Pg = (self.P_min + self.P_max) / 2.0
        self.intended_Pg = midpoint_Pg
        safe_pg = self._apply_safety_layer(midpoint_Pg, self.current_Pd)
        self.current_Pg = np.array(safe_pg).flatten()
        
        # --- Clear frames for a new episode ---
        self.frames = []
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Calculate current line loading as part of the state
        pg_vec = vec(self.current_Pg, self.n_gens)
        pd_vec = vec(self.current_Pd, self.n_buses)
        line_flows = self.PTDF @ (self.M @ pg_vec - pd_vec)
        loading_pct = np.tanh(np.abs(line_flows) / self.F_max)
        
        # Concatenate: [Gen Power (4), Load Demand (4), Line Loading (4)]
        obs = np.concatenate([
            pg_vec, 
            pd_vec, 
            loading_pct
        ]).astype(np.float32)
        return obs

    def _apply_safety_layer(self, Pg_rl, Pd):
        """The QP Projection Step (Constraint Handling)"""
        Pg = cp.Variable(self.n_gens)
        
        # Minimize the Euclidean distance between agent suggestion and Safe reality
        objective = cp.Minimize(cp.sum_squares(Pg - Pg_rl))
        
        # Constraints using the Universal PTDF logic
        constraints = [
            cp.sum(Pg) == cp.sum(Pd),
            Pg >= self.P_min,                              
            Pg <= self.P_max,
            self.PTDF @ (self.M @ Pg - Pd) <= self.F_max,  # Thermal Limits
            self.PTDF @ (self.M @ Pg - Pd) >= -self.F_max
        ]
        
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(
                solver=cp.OSQP,
                verbose=False,
                warm_start=True,
                eps_abs=1e-5,
                eps_rel=1e-5,
                max_iter=5000
            )
        except:
            clipped = np.clip(Pg_rl, self.P_min, self.P_max)
            scale = np.sum(Pd) / max(np.sum(clipped), 1e-6)
            return clipped * scale
        
        if Pg.value is None or np.any(np.isnan(Pg.value)):
            Pg_safe = np.clip(Pg_rl, self.P_min, self.P_max)
        
            # enforce power balance approximately
            total_load = np.sum(Pd)
            total_gen = np.sum(Pg_safe)
        
            if total_gen > 1e-6:
                Pg_safe = Pg_safe * (total_load / total_gen)
        
            # re-clip after scaling
            Pg_safe = np.clip(Pg_safe, self.P_min, self.P_max)
        
            return vec(Pg_safe, self.n_gens)
            
        return vec(Pg.value, self.n_gens)

    def step(self, action):
        # Update Episode Clock
        self.current_step += 1
        
        action = vec(action, self.n_gens)
        
        # Sequential Load Dynamics (Drift)
        drift = self.np_random.uniform(-self.load_drift, self.load_drift, size=self.n_buses)
        self.current_Pd = np.clip(self.current_Pd + drift, self.P_d_base * 0.8, self.P_d_base * 1.2)
    
        # Map Agent Intended Action [-1, 1] to Physical Limits [P_min, P_max]
        rescaled_Pg = (self.P_min + (action + 1.0) * 0.5 * (self.P_max - self.P_min)).flatten()
        
            
        self.intended_Pg = rescaled_Pg
            
        # --- Calculate Raw Violation (Agent's Intent) ---
        raw_flows = self.PTDF @ (self.M @ rescaled_Pg - self.current_Pd)
        raw_violation = 100 * np.sum(np.maximum(0, np.abs(raw_flows) - 0.95*self.F_max))
        
        # Apply Safety Projection
        self.current_Pg = self._apply_safety_layer(rescaled_Pg, self.current_Pd)
        
        # Physics Calculation (Final Line Flows)
        line_flows = self.PTDF @ (self.M @ self.current_Pg - self.current_Pd)
        violation = 100 * np.sum(np.maximum(0, np.abs(line_flows) - self.F_max))
        
        # Calculate Loading Percentage for the Dashboard
        loading_pcts = (np.abs(line_flows) / self.F_max) * 100
        max_pct = np.max(loading_pcts)
        
        # --- Economic Cost Calculation (For Logging/Render) ---
        a_coeffs = np.array([0.1085, 0.1085, 0.1085, 0.1085])
        b_coeffs = np.array([0.0832, 0.0260, 0.0832, 0.0260])
        c_coeffs = np.array([0.008, 0.006, 0.008, 0.006])
        cost_multiplier = 5000.0  # To ensure info['cost'] is in USD
        
        # Calculate Intended Total Cost ($/hr)
        intended_cost = np.sum(a_coeffs * self.intended_Pg**2 + b_coeffs * self.intended_Pg + c_coeffs) * cost_multiplier
        
        # Calculate Actual Total Cost ($/hr)
        actual_cost = np.sum(a_coeffs * self.current_Pg**2 + b_coeffs * self.current_Pg + c_coeffs) * cost_multiplier
        
        # --- Reward Calculation (Optimized for TD3 Stability) ---
        intended_total_mw = np.sum(self.intended_Pg) * 100
        actual_total_mw = np.sum(self.current_Pg) * 100
        mismatch = np.abs(intended_total_mw - actual_total_mw)
        
        reward_cost = -np.tanh(actual_cost / 1000.0)   # Scaled for Critic stability
        penalty_violation = -np.tanh(raw_violation / 50.0)
        redispatch_penalty = -5 * np.linalg.norm(self.current_Pg - self.intended_Pg, 1)
        
        reward = reward_cost + penalty_violation + redispatch_penalty
        
        # Lifecycle Management
        obs = self._get_obs()
        terminated = False 
        truncated = self.current_step >= self.max_steps
        
        info = {
            "step": self.current_step,
            "intended_cost": intended_cost,
            "actual_cost": actual_cost,
            "violation": violation,         # Post-Safety
            "raw_violation": raw_violation, # Pre-Safety
            "max_loading_pct": max_pct,
            "line_flows": line_flows,
            "Pg_intended": self.intended_Pg,
            "Pg_safe": self.current_Pg,
            "Pd_current": self.current_Pd,
            "mismatch": mismatch,
            "cost": actual_cost
        }
        
        self.intended_cost = intended_cost 
        self.actual_cost = actual_cost
        
        return obs, reward, terminated, truncated, info

    def render(self, mode="human", episode_reward=0.0):
        S_base = 100.0  
        
        line_flows = self.PTDF @ (self.M @ self.current_Pg - self.current_Pd)
        loading_pct = (np.abs(line_flows) / self.F_max) * 100
        
        intended_hourly_cost = self.intended_cost 
        actual_hourly_cost = self.actual_cost
        
        # Efficiency Metrics
        total_load_mw = np.sum(self.current_Pd) * S_base
        avg_rate = actual_hourly_cost / total_load_mw if total_load_mw > 0 else 0

        plt.close('all') 
        fig = plt.figure(figsize=(18, 9), facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1], wspace=0.2)
        ax0 = fig.add_subplot(gs[0, 0]) 
        ax1 = fig.add_subplot(gs[0, 1]) 
        
        ax0.set_title(f"4-Bus System Network (case4gs) | Step: {self.current_step}/{self.max_steps}", 
                      fontsize=22, fontweight='black', color='#2c3e50', pad=25)
        
        ax0.set_xlim(-3.5, 7.5)
        ax0.set_ylim(-2.0, 6.0) 
        ax0.axis('off')

        # --- TD3 OPERATOR STATUS ---
        ax0.add_patch(plt.Circle((-2.5, 5.4), 0.4, color='#3498db', alpha=0.3, zorder=10))
        ax0.text(-2.5, 5.4, "🤖", fontsize=22, ha='center', va='center', zorder=11)
        ax0.text(-1.9, 5.7, "AGENT OPERATOR INTERFACE", fontsize=10, fontweight='bold', color='#34495e')
        
        # Logic to determine the Agent's "thought"
        if np.any(loading_pct > 95): status_msg = "CRITICAL: REROUTING FLOW"
        elif np.any(loading_pct > 75): status_msg = "WARNING: CONGESTION MGMT"
        else: status_msg = "MODE: ECONOMIC DISPATCH"
        
        ax0.text(-1.9, 5.4, status_msg, fontsize=11, fontweight='black', 
                 color='#c0392b' if "CRITICAL" in status_msg else '#27ae60',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='#bdc3c7', boxstyle='round,pad=0.3'))

        # --- Show the total demand and supply ---
        total_supply_mw = np.sum(self.current_Pg) * S_base
        total_demand_mw = np.sum(self.current_Pd) * S_base
        
        # Add a Balance Summary box
        ax0.text(1.0, 5.4, f"TOTAL DEMAND: {total_demand_mw:.1f} MW", fontsize=10, fontweight='bold', color='#2980b9')
        ax0.text(1.0, 5.7, f"TOTAL SUPPLY: {total_supply_mw:.1f} MW", fontsize=10, fontweight='bold', color='#27ae60')
        
        # --- Draw Buses & Symbols ---
        bus_coords = [(0, 4), (0, 0), (5, 4), (5, 0)] 
        for i, (x, y) in enumerate(bus_coords):
            ax0.add_patch(plt.Rectangle((x-0.1, y-0.6), 0.2, 1.2, color='#2c3e50', zorder=5))
            ax0.text(x, y+0.8, f"BUS {i+1}", fontsize=12, fontweight='black', ha='center')
            
            gens_at_this_bus = [idx for idx, b_idx in enumerate(self.gen_bus_map) if b_idx == i]
            for offset_idx, g_idx in enumerate(gens_at_this_bus):
                g_val = self.current_Pg[g_idx] * S_base
                if g_val > 0.01:
                    y_pos = y + (0.35 - offset_idx*0.7) if len(gens_at_this_bus) > 1 else y + 0.35
                    ax0.add_patch(plt.Circle((x + 0.4, y_pos), 0.18, color='#27ae60', alpha=0.9, zorder=6))
                    ax0.text(x + 0.7, y_pos, f"G{g_idx+1}: {g_val:.1f} MW", color='#27ae60', fontsize=10, fontweight='bold', va='center')

            nodal_load_mw = self.current_Pd[i] * S_base
            if nodal_load_mw > 0.1:
                tri_pts = np.array([[x-0.3, y-0.1], [x-0.5, y-0.4], [x-0.1, y-0.4]])
                ax0.add_patch(plt.Polygon(tri_pts, color='#2980b9', alpha=0.9, zorder=6))
                ax0.text(x - 1.6, y - 0.45, f"L: {nodal_load_mw:.1f} MW", color='#2980b9', fontsize=10, fontweight='bold', ha='left')

        # --- Transmission Lines ---
        for i, (f, t, _) in enumerate(self.lines_data):
            f_c, t_c = bus_coords[f], bus_coords[t]
            pct = loading_pct[i]
            color = '#c0392b' if pct > 100 else ('#f1c40f' if pct > 70 else '#27ae60')
            ax0.plot([f_c[0], t_c[0]], [f_c[1], t_c[1]], color=color, linewidth=3, alpha=0.6, zorder=1)
            mx, my = (f_c[0]+t_c[0])/2, (f_c[1]+t_c[1])/2
            direction = '→' if line_flows[i] > 0 else '←'
            ax0.text(mx, my, f"{direction} {pct:.1f}%", fontsize=11, fontweight='black', 
                     ha='center', color='white', bbox=dict(facecolor=color, alpha=1.0, edgecolor='none', boxstyle='round,pad=0.3'))

        # --- Performance Labels & Legend ---
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60', markersize=10, label='Generator'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#2980b9', markersize=10, label='Load'),
            Patch(facecolor='#f1c40f', label='Congestion Warning (70-100%)'),
            Patch(facecolor='#c0392b', label='Line Overload (>100%)')
        ]
        ax0.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.0, 0.03), fontsize=10, frameon=True, shadow=True)

        # --- Side-by-Side Dispatch Panel ---
        ax1.set_title("Real-Time Dispatch (MW)\n(Agent Intent vs. Physical Safety)", 
                     fontsize=15, fontweight='black', pad=15)
        gen_colors = ['#1abc9c', '#e67e22', '#9b59b6', '#e91e63'] 
        
        bottom_intended = 0
        bottom_actual = 0
        width = 0.75
        x_positions = [0.6, 1.7]
        labels = ['Intended\n(Agent)', 'Actual\n(Safe)']

        for i in range(self.n_gens):
            intended_mw = self.intended_Pg[i] * S_base
            actual_mw = self.current_Pg[i] * S_base
            
            # Plot Intended Bar (Agent's Intent)
            ax1.bar(x_positions[0], intended_mw, width, bottom=bottom_intended, 
                    color=gen_colors[i], alpha=0.75, edgecolor='black', linestyle='--')
            
            # Label Intended
            if intended_mw > 25:
                ax1.text(x_positions[0], bottom_intended + intended_mw/2, f"G{i+1}\n{intended_mw:.1f}", 
                         ha='center', va='center', color='white', fontweight='bold', fontsize=11)

            # Plot Actual Bar (Physical Safety)
            ax1.bar(x_positions[1], actual_mw, width, bottom=bottom_actual, 
                    color=gen_colors[i], alpha=1.0, edgecolor='black')
            
            # Label Actual
            if actual_mw > 25:
                ax1.text(x_positions[1], bottom_actual + actual_mw/2, f"G{i+1}\n{actual_mw:.1f}", 
                         ha='center', va='center', color='white', fontweight='bold', fontsize=11)
            
            bottom_intended += intended_mw
            bottom_actual += actual_mw

        # Add Total Labels on top
        ax1.text(x_positions[0], bottom_intended + 8, f"TOTAL\n{bottom_intended:.1f}", 
                 ha='center', fontweight='bold', color='#7f8c8d', fontsize=11)
        ax1.text(x_positions[1], bottom_actual + 8, f"TOTAL\n{bottom_actual:.1f}", 
                 ha='center', fontweight='bold', color='#2c3e50', fontsize=11)

        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(labels, fontsize=11, fontweight='bold')
        ax1.set_xlim(-0.2, 2.5)
        ax1.set_ylim(0, 650) 
        ax1.grid(axis='y', linestyle=':', alpha=0.5)

        # Footer
        footer_text = (f"INTENDED: ${intended_hourly_cost:,.2f}/hr  |  "
                   f"ACTUAL: ${actual_hourly_cost:,.2f}/hr  |  "
                   f"RATE: ${avg_rate:.2f}/MWh\n"
                   f"TOTAL REWARD: {episode_reward:,.1f}")

        fig.text(0.45, 0.04, footer_text, ha='center', fontsize=14, fontweight='black',
                 bbox=dict(facecolor='#f1c40f', alpha=0.9, edgecolor='#f39c12', boxstyle='round,pad=0.5'))

        # --- ANIMATION LOGIC ---
        if mode == "human" or mode == "rgb_array":
            if mode == "human":
                clear_output(wait=True)
                display(fig)
                
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=120, facecolor='white')
            buf.seek(0)
            
            # Convert to numpy array for video/GIF processing
            img = np.array(Image.open(buf).convert('RGB'))
            self.frames.append(img)
            
            # Cleanup to avoid multiple render plots
            plt.close(fig) 
            if mode == "human": 
                time.sleep(0.5)
        else:
            # For non-animation modes 
            plt.show()

if __name__ == "__main__":
    env = PowerSystemEnv()
    print("Environment defined successfully.")