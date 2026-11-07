# AegisAI Advanced Modern GUI
# Adaptive Multimodal AI for Personalized Healthcare & Wellness Risk Assessment
#
# Training architecture optimized for large tabular datasets (~300K rows):
# - Leakage-safe preprocessing: fit ONLY on training data.
# - Train/validation/test = 70/15/15.
# - Validation is used for early stopping + model selection.
# - Test set remains untouched until the final selected model is evaluated.
# - Strong regularization + conservative tree depth + subsampling.
# - Early stopping for XGBoost, LightGBM, CatBoost, HistGradientBoosting and ANN.
# - Logistic Regression uses regularization and class balancing.
# - Train/validation gap is reported as an overfitting diagnostic.
#
# IMPORTANT:
# No ML architecture can mathematically guarantee "zero overfitting".
# This trainer substantially reduces overfitting risk and exposes it explicitly.

import os
import sys
import threading
import queue
import traceback
import time
import warnings
import subprocess
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import joblib

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

# Optional large-tabular models.
# The GUI will continue to work if one optional package is not installed.
try:
    from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False


APP_TITLE = "AegisAI • Adaptive Multimodal Healthcare AI"
DEFAULT_DATASET = r"C:\Users\mayan\Desktop\GENai\AegisAI_Multimodal_Health_Wellness_Risk_Dataset_300K_Synthetic.csv"
MODEL_PATH = "AegisAI_Health_Risk_Model.joblib"
WALLPAPER_NAME = "BgWallpaper.png"
SOURCE_GUI_FILE = "V3_AegisAI_GUI_Adaptive_Multimodal_Updated.py"

TARGET = "Risk_Level"
DROP_COLUMNS = [TARGET, "Person_ID", "Wellness_Score_0_100"]

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE_FROM_REMAINING = 0.17647058823529413  # 15% of total => 0.17647 of remaining 85%
MAX_DATASET_ROWS = 300_000

# Model-selection guardrail. A model with a very large train->validation
# Macro-F1 gap is considered suspiciously overfit.
# General overfitting guardrails.
# 0–2% is the preferred zone; 5% is still generally usable;
# >10% is treated as a serious warning.
OVERFIT_GAP_GOOD = 0.02
OVERFIT_GAP_WARN = 0.05
OVERFIT_GAP_HARD = 0.10

BG = "#08111F"
PANEL = "#0E1A2B"
CARD = "#14253C"
PANEL2 = "#111F33"
BORDER = "#213752"
TEXT = "#EAF2FF"
MUTED = "#8FA6C1"
ACCENT = "#19D3AE"
BLUE = "#4F8CFF"
RED = "#FF5C7A"
YELLOW = "#F5B942"


class BackgroundTerminal(tk.Frame):
    """Canvas-based terminal with a wallpaper background and scrolling support."""
    def __init__(self, master, wallpaper_path=None, **kwargs):
        super().__init__(master, bg="#050B14", **kwargs)
        self.lines = []
        self.font = ("Consolas", 12)
        self.line_height = 22
        self.pad_x = 12
        self.pad_y = 12
        self._bg_photo = None
        self._bg_original = None
        self._bg_path = wallpaper_path

        self.canvas = tk.Canvas(
            self, bg="#050B14", highlightthickness=0, bd=0,
            xscrollincrement=1, yscrollincrement=1
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

        if PIL_AVAILABLE and wallpaper_path and os.path.exists(wallpaper_path):
            try:
                self._bg_original = Image.open(wallpaper_path).convert("RGB")
            except Exception:
                self._bg_original = None

    def _on_resize(self, _event=None):
        self._draw_background()
        self._redraw()

    def _draw_background(self):
        if self._bg_original is None:
            self._bg_photo = None
            self.canvas.configure(bg="#050B14")
            return
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        iw, ih = self._bg_original.size
        scale = max(w / iw, h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = self._bg_original.resize((nw, nh), Image.Resampling.LANCZOS)
        left = max(0, (nw - w) // 2)
        top = max(0, (nh - h) // 2)
        img = img.crop((left, top, left + w, top + h))
        self._bg_photo = ImageTk.PhotoImage(img)
        self.canvas.delete("background")
        self.canvas.create_image(0, 0, image=self._bg_photo, anchor="nw", tags="background")
        self.canvas.tag_lower("background")

    def _redraw(self):
        self.canvas.delete("terminal_text")
        max_width = max(self.canvas.winfo_width(), 900)
        for i, line in enumerate(self.lines):
            self.canvas.create_text(
                self.pad_x,
                self.pad_y + i * self.line_height,
                anchor="nw",
                text=line,
                fill="#00FF66",
                font=self.font,
                tags="terminal_text",
            )
        content_h = max(self.canvas.winfo_height(), self.pad_y * 2 + len(self.lines) * self.line_height)
        longest = max([len(x) for x in self.lines], default=1)
        content_w = max(max_width, self.pad_x * 2 + longest * 7.3)
        self.canvas.configure(scrollregion=(0, 0, content_w, content_h))
        self.canvas.tag_raise("terminal_text")

    def insert(self, index, text):
        parts = str(text).splitlines()
        if not parts:
            return
        if index in ("end", "insert"):
            self.lines.extend(parts)
        else:
            self.lines.extend(parts)
        self._redraw()

    def delete(self, *_args):
        self.lines.clear()
        self._redraw()

    def see(self, _index="end"):
        self.canvas.yview_moveto(1.0)

    def configure(self, **kwargs):
        ycmd = kwargs.pop("yscrollcommand", None)
        xcmd = kwargs.pop("xscrollcommand", None)
        if ycmd:
            self.canvas.configure(yscrollcommand=ycmd)
        if xcmd:
            self.canvas.configure(xscrollcommand=xcmd)
        if kwargs:
            super().configure(**kwargs)

    config = configure

    def yview(self, *args):
        return self.canvas.yview(*args)

    def xview(self, *args):
        return self.canvas.xview(*args)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class AegisAIApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1500x900")
        self.minsize(1180, 740)
        self.configure(bg=BG)

        self.q = queue.Queue()
        self.training = False

        self.df = None
        self.results_df = None
        self.best_model = None
        self.best_model_metrics = None
        self.best_model_name = None

        # Cumulative training state:
        # previously trained models remain available during this GUI session.
        self.trained_models = {}
        self.model_predictions = {}
        self.model_metrics = {}
        self.model_preprocessors = {}
        self.model_label_encoders = {}
        self.model_features = {}
        self.training_dataset_path = None

        self.preprocessor = None
        self.label_encoder = None
        self.features = []
        self.num_features = []
        self.cat_features = []

        self.training_started_at = None
        self.training_session_started_at = None
        self.report_text = ""
        self.animation_phase = 0

        self.dataset = tk.StringVar(value=DEFAULT_DATASET)
        self.status = tk.StringVar(value="Ready")
        self.best = tk.StringVar(value="—")
        self.acc = tk.StringVar(value="—")
        self.f1 = tk.StringVar(value="—")
        self.bacc = tk.StringVar(value="—")
        self.samples = tk.StringVar(value="—")
        self.gap = tk.StringVar(value="—")

        self.styles()
        self.build()

        self.after(100, self.events)

        self.log("=" * 72)
        self.log("AEGISAI Advanced Control Center Ready")
        self.log("Large-tabular training architecture: train/validation/test = 70/15/15")
        self.log("Model selection: validation Macro F1 with overfitting-gap diagnostics")
        self.log("Click 'Train All Models' or use an individual model button on Dashboard.")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def exit_app(self):
        """Safely close the application."""
        if self.training:
            if not messagebox.askyesno(
                "Exit AegisAI",
                "Model training is still running. Do you really want to exit?",
            ):
                return
        self.destroy()

    def styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(
            "Treeview",
            background=PANEL2,
            foreground=TEXT,
            fieldbackground=PANEL2,
            rowheight=32,
            font=("Segoe UI", 9),
        )
        s.configure(
            "Treeview.Heading",
            background="#172A43",
            foreground=TEXT,
            font=("Segoe UI Semibold", 9),
        )
        s.map("Treeview", background=[("selected", "#1E4B66")])
        s.configure(
            "TProgressbar",
            troughcolor="#172A43",
            background=ACCENT,
        )

    def btn(self, p, text, cmd, color=BLUE, border_color=None, border_width=0):
        fg = "#06151A" if color == ACCENT else "#fff"

        return tk.Button(
            p,
            text=text,
            command=cmd,
            bg=color,
            fg=fg,
            activebackground=color,
            activeforeground=fg,
            relief="solid" if border_width > 0 else "flat",
            bd=border_width,
            highlightthickness=border_width,
            highlightbackground=border_color if border_color else color,
            highlightcolor=border_color if border_color else color,
            cursor="hand2",
            font=("Segoe UI Semibold", 10),
            padx=10,
            pady=9,
        )

    def card(self, p, title, var, color):
        f = tk.Frame(
            p,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        f.pack(side="left", fill="both", expand=True, padx=4)
        tk.Label(
            f,
            text=title,
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", padx=12, pady=(10, 2))
        tk.Label(
            f,
            textvariable=var,
            bg=CARD,
            fg=color,
            font=("Segoe UI Semibold", 16),
        ).pack(anchor="w", padx=12, pady=(0, 10))

    def build(self):
        h = tk.Frame(self, bg=BG, height=76)
        h.pack(fill="x", padx=22, pady=(15, 5))
        h.pack_propagate(False)

        x = tk.Frame(h, bg=BG)
        x.pack(side="left")

        tk.Label(
            x,
            text="✦ AegisAI : Adaptive Multimodal AI • Healthcare & Wellness Risk Assessment",
            bg=BG,
            fg=ACCENT,
            font=("Segoe UI Black", 25),
        ).pack(anchor="w")

        tk.Label(
            x,
            text="Developed By MAYANK SINGH ✅ ",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        r = tk.Frame(h, bg=BG)
        r.pack(side="right")

        # Top-right application exit button
        self.btn(r, "✕  EXIT", self.exit_app, RED, border_color="#FFFFFF", border_width=1).pack(side="right", padx=(12, 0), pady=(0, 8))

        tk.Label(
            r,
            textvariable=self.status,
            bg=BG,
            fg=ACCENT,
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="e")

        self.pb = ttk.Progressbar(r, mode="indeterminate", length=220)
        self.pb.pack(anchor="e", pady=5)

        self.anim_label = tk.Label(
            r,
            text="● READY",
            bg=BG,
            fg=ACCENT,
            font=("Segoe UI Semibold", 9),
        )
        self.anim_label.pack(anchor="e")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=22, pady=8)

        side = tk.Frame(
            body,
            bg=PANEL,
            width=235,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        side.pack(side="left", fill="y", padx=(0, 12))
        side.pack_propagate(False)

        tk.Label(
            side,
            text="CONTROL CENTER",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=16, pady=(20, 10))

        self.btn(side, "▶  Train All Models", self.start, "#19D3AE").pack(
            fill="x", padx=12, pady=4
        )
        self.btn(side, "◉  Load Model", self.load, "#4F8CFF").pack(
            fill="x", padx=12, pady=4
        )
        self.btn(side, "▣  Save Best Model", self.save, "#8B5CF6").pack(
            fill="x", padx=12, pady=4
        )
        self.btn(side, "✕  Delete Saved Model", self.delete, "#DB1438").pack(
            fill="x", padx=12, pady=4
        )
        self.btn(side, "↻  Clear Dashboard", self.clear, "#F59E0B").pack(
            fill="x", padx=12, pady=4
        )
        self.btn(side, "📄  Save Report", self.save_report, "#06B6D4").pack(
            fill="x", padx=12, pady=4
        )
        self.btn(side, "📊  Export Metrics CSV", self.export_csv, "#EC4899").pack(
            fill="x", padx=12, pady=4
        )
        self.btn(
            side,
            "💾  Save Individual Model",
            self.save_individual_model,
            "#F97316",
        ).pack(fill="x", padx=12, pady=4)
        self.btn(side, "🚀  Run AegisAI", self.open_v3_file, "#6366F1").pack(
            fill="x", padx=12, pady=4
        )

        ttk.Separator(side).pack(fill="x", padx=16, pady=16)

        tk.Label(
            side,
            text="DATASET",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=16)

        self.btn(side, "⌕  Select Dataset", self.select, "#14B8A6").pack(
            fill="x", padx=12, pady=5
        )

        tk.Label(
            side,
            textvariable=self.dataset,
            bg=PANEL,
            fg=MUTED,
            wraplength=195,
            justify="left",
            font=("Segoe UI", 8),
        ).pack(anchor="w", padx=16, pady=6)

        tk.Label(
            side,
            text=(
                "Selection metric: Validation Macro F1\n"
                "Test set is untouched until final evaluation\n"
                "Overfit gap = Train F1 − Validation F1"
            ),
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
            justify="left",
        ).pack(side="bottom", anchor="w", padx=16, pady=18)

        main = tk.Frame(body, bg=BG)
        main.pack(side="left", fill="both", expand=True)

        nb = ttk.Notebook(main)
        nb.pack(fill="both", expand=True)

        self.dash = tk.Frame(nb, bg=BG)
        self.con = tk.Frame(nb, bg=BG)
        self.pred = tk.Frame(nb, bg=BG)

        nb.add(self.dash, text="  Dashboard  ")
        nb.add(self.con, text="  Live Terminal  ")
        nb.add(self.pred, text="  Prediction  ")

        self.nb = nb

        self.build_dash()
        self.build_console()
        self.build_pred()

    def build_console(self):
        """Build the live training terminal/log panel."""
        outer = tk.Frame(
            self.con,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        outer.pack(fill="both", expand=True)

        header = tk.Frame(outer, bg=PANEL)
        header.pack(fill="x", padx=14, pady=(12, 8))

        tk.Label(
            header,
            text="LIVE TRAINING TERMINAL",
            bg=PANEL,
            fg=ACCENT,
            font=("Segoe UI Semibold", 12),
        ).pack(side="left")

        tk.Button(
            header,
            text="Clear Console",
            command=lambda: self.console.delete("1.0", "end"),
            bg="#111F33",
            fg=TEXT,
            activebackground="#1A304A",
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=BORDER,
            cursor="hand2",
            font=("Segoe UI", 9),
            padx=10,
            pady=5,
        ).pack(side="right")

        wallpaper_candidates = [
            Path(__file__).with_name(WALLPAPER_NAME),
            Path.cwd() / WALLPAPER_NAME,
        ]
        wallpaper_path = next((str(x) for x in wallpaper_candidates if x.exists()), None)

        terminal_frame = tk.Frame(outer, bg="#050B14")
        terminal_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.console = BackgroundTerminal(
            terminal_frame,
            wallpaper_path=wallpaper_path,
        )
        if wallpaper_path:
            self.log(f"Terminal wallpaper loaded: {WALLPAPER_NAME}")
        else:
            self.log(f"Terminal wallpaper not found: {WALLPAPER_NAME} (using dark fallback)")

        console_scroll = ttk.Scrollbar(
            terminal_frame,
            orient="vertical",
            command=self.console.yview,
        )
        console_hscroll = ttk.Scrollbar(
            terminal_frame,
            orient="horizontal",
            command=self.console.xview,
        )
        self.console.configure(
            yscrollcommand=console_scroll.set,
            xscrollcommand=console_hscroll.set,
        )

        self.console.pack(side="left", fill="both", expand=True)
        console_scroll.pack(side="right", fill="y")
        console_hscroll.pack(side="bottom", fill="x")

    def build_dash(self):
        c = tk.Frame(self.dash, bg=BG)
        c.pack(fill="x", pady=(0, 10))

        self.card(c, "BEST MODEL", self.best, ACCENT)
        self.card(c, "TEST ACCURACY", self.acc, BLUE)
        self.card(c, "TEST MACRO F1", self.f1, YELLOW)
        self.card(c, "TEST BALANCED ACC.", self.bacc, ACCENT)
        self.card(c, "OVERFIT GAP", self.gap, RED)
        self.card(c, "DATASET ROWS", self.samples, TEXT)

        box = tk.Frame(
            self.dash,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        box.pack(fill="both", expand=True)

        tk.Label(
            box,
            text="Model Comparison • 300K-Scale Tabular Architecture",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 13),
        ).pack(anchor="w", padx=15, pady=(12, 6))

        # Individual model training controls.
        # Train All Models remains in the left control panel.
        train_panel = tk.Frame(
            box,
            bg="#0B1727",
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        train_panel.pack(fill="x", padx=12, pady=(0, 10))

        tk.Label(
            train_panel,
            text="TRAIN A SELECTED MODEL",
            bg="#0B1727",
            fg=MUTED,
            font=("Segoe UI Semibold", 15),
        ).pack(anchor="w", padx=12, pady=(9, 6))

        model_button_frame = tk.Frame(train_panel, bg="#0B1727")
        model_button_frame.pack(fill="x", padx=10, pady=(0, 10))

        individual_buttons = [
            ("Logistic Regression", "Train Logistic Regression", "#000000", "#FFFFFF"),
            ("Random Forest", "Train Random Forest", "#000000", "#FFFFFF"),
            ("Extra Trees", "Train Extra Trees", "#000000", "#FFFFFF"),
            ("ANN", "Train ANN", "#000000", "#FFFFFF"),
            ("XGBoost", "Train XGBoost", "#000000", "#FFFFFF"),
            ("LightGBM", "Train LightGBM", "#000000", "#FFFFFF"),
            ("CatBoost", "Train CatBoost", "#000000", "#FFFFFF"),
            ("HistGradientBoosting", "Train HistGradientBoosting", "#000000", "#FFFFFF"),
        ]

        self.individual_train_buttons = {}
        for idx, (model_name, label, color, border_color) in enumerate(individual_buttons):
            b = self.btn(
                model_button_frame,
                "▶  " + label,
                lambda m=model_name: self.start_single(m),
                color,
                border_color=border_color,
                border_width=2,
            )
            b.grid(
                row=idx // 4,
                column=idx % 4,
                sticky="ew",
                padx=4,
                pady=4,
            )
            self.individual_train_buttons[model_name] = b

        for col in range(4):
            model_button_frame.grid_columnconfigure(col, weight=1)

        wrap = tk.Frame(box, bg=PANEL)
        wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        cols = (
            "Model",
            "Train F1",
            "Val F1",
            "Test F1",
            "Val Acc.",
            "Test Acc.",
            "Overfit Gap",
            "Status",
        )

        self.table = ttk.Treeview(
            wrap,
            columns=cols,
            show="headings",
        )

        widths = (185, 90, 90, 90, 95, 95, 105, 100)

        for c, w in zip(cols, widths):
            self.table.heading(c, text=c)
            self.table.column(c, width=w, anchor="center")

        sc = ttk.Scrollbar(wrap, command=self.table.yview)
        self.table.configure(yscrollcommand=sc.set)

        self.table.pack(side="left", fill="both", expand=True)
        sc.pack(side="right", fill="y")

        self.table.tag_configure(
            "winner",
            background="#143B3A",
            foreground=ACCENT,
        )
        self.table.tag_configure(
            "warning",
            background="#3A3214",
            foreground=YELLOW,
        )
        self.table.tag_configure(
            "overfit",
            background="#401B27",
            foreground=RED,
        )

    def build_pred(self):
        tk.Label(   
            self.pred,
            text="Health Risk Prediction",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI Semibold", 14),
        ).pack(anchor="w", pady=(0, 10))

        left = tk.Frame(
            self.pred,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = tk.Frame(
            self.pred,
            bg=PANEL,
            width=310,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        fields = [
            ("Age", "22"),
            ("Sex", "Male"),
            ("Height_cm", "175"),
            ("Weight_kg", "68"),
            ("BMI", "22.2"),
            ("Stress_Level_0_10", "3"),
            ("Mood_Score_0_10", "8"),
            ("Heart_Rate_bpm", "68"),
            ("SpO2_percent", "98"),
            ("Sleep_Hours", "7.5"),
            ("Daily_Steps", "8500"),
            ("Exercise_Minutes_Week", "180"),
            ("AQI", "80"),
            ("Humidity_percent", "55"),
            ("UV_Index", "5"),
        ]

        self.pvars = {}

        numeric = {
            "Age",
            "Height_cm",
            "Weight_kg",
            "BMI",
            "Stress_Level_0_10",
            "Mood_Score_0_10",
            "Heart_Rate_bpm",
            "SpO2_percent",
            "Sleep_Hours",
            "Daily_Steps",
            "Exercise_Minutes_Week",
            "AQI",
            "Humidity_percent",
            "UV_Index",
        }

        for i, (name, val) in enumerate(fields):
            row = 1 + i // 2
            col = (i % 2) * 2

            tk.Label(
                left,
                text=name,
                bg=PANEL,
                fg=MUTED,
                font=("Segoe UI", 9),
            ).grid(
                row=row,
                column=col,
                sticky="w",
                padx=(16, 7),
                pady=7,
            )

            v = tk.StringVar(value=val)
            self.pvars[name] = v

            tk.Entry(
                left,
                textvariable=v,
                bg="#0A1524",
                fg=TEXT,
                insertbackground=TEXT,
                relief="flat",
                font=("Segoe UI", 9),
                width=17,
            ).grid(
                row=row,
                column=col + 1,
                sticky="ew",
                padx=(0, 16),
                pady=7,
            )

        self.btn(
            left,
            "Run Risk Prediction",
            self.predict,
            ACCENT,
        ).grid(row=9, column=0, columnspan=4, pady=18)

        tk.Label(
            right,
            text="ASSESSMENT",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=20, pady=(24, 7))

        self.risk = tk.Label(
            right,
            text="—",
            bg=PANEL,
            fg=ACCENT,
            font=("Segoe UI Black", 32),
        )
        self.risk.pack(anchor="w", padx=20)

        self.pdetail = tk.Label(
            right,
            text="Train or load a model first.",
            bg=PANEL,
            fg=MUTED,
            justify="left",
            wraplength=260,
            font=("Segoe UI", 9),
        )
        self.pdetail.pack(anchor="w", padx=20, pady=12)

    # ------------------------------------------------------------------
    # Event / logging
    # ------------------------------------------------------------------

    def log(self, s):
        # Timestamp every terminal entry so long training runs are easy to track.
        timestamp = time.strftime("%H:%M:%S")
        self.q.put(("log", f"[{timestamp}] {str(s)}"))

    def events(self):
        try:
            while True:
                typ, val = self.q.get_nowait()

                if typ == "log":
                    self.console.insert("end", val + "\n")
                    self.console.see("end")

                elif typ == "done":
                    total_elapsed = (
                        time.perf_counter() - self.training_session_started_at
                        if getattr(self, "training_session_started_at", None) is not None
                        else 0.0
                    )
                    self.log(
                        f"TOTAL TRAINING SESSION TIME: "
                        f"{total_elapsed:.2f}s ({total_elapsed/60:.2f} min)"
                    )
                    self.training = False
                    self.pb.stop()
                    self._set_training_buttons_state(True)
                    self.status.set("Training complete")
                    self.anim_label.config(text="● READY", fg=ACCENT)
                    self.update_dash()
                    self.nb.select(self.dash)

                    gap = self.best_model_metrics.get(
                        "Overfit Gap",
                        0,
                    )

                    messagebox.showinfo(
                        "Training Complete",
                        (
                            f"Training finished.\n\n"
                            f"Current best model: {self.best_model_name}\n"
                            f"Best Test Macro F1: "
                            f"{self.best_model_metrics['Test Macro F1']:.4f}\n"
                            f"Best Overfit gap: {gap:.4f}\n\n"
                            f"All previously trained models are retained in memory.\n"
                            f"Production model saved: "
                            f"{os.path.abspath(MODEL_PATH)}"
                        ),
                    )

                elif typ == "error":
                    self.training = False
                    self.pb.stop()
                    self._set_training_buttons_state(True)
                    self.status.set("Training failed")
                    self.anim_label.config(text="● ERROR", fg=RED)
                    messagebox.showerror("Training Failed", val)

        except queue.Empty:
            pass

        self.after(100, self.events)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def select(self):
        f = filedialog.askopenfilename(
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ]
        )

        if f:
            self.dataset.set(f)
            self.log("Dataset selected: " + f)

    @staticmethod
    def _safe_stratified_cap(X, y, max_rows=MAX_DATASET_ROWS):
        """Keep a maximum number of rows while preserving class proportions."""
        if len(X) <= max_rows:
            return X, y

        tmp = pd.DataFrame({"__target__": y})
        _, sample_idx = train_test_split(
            np.arange(len(tmp)),
            test_size=max_rows,
            random_state=RANDOM_STATE,
            stratify=tmp["__target__"],
        )

        sample_idx = np.sort(sample_idx)
        return X.iloc[sample_idx].reset_index(drop=True), y.iloc[sample_idx].reset_index(drop=True)

    @staticmethod
    def _make_encoder():
        # sklearn 1.2+ uses sparse_output. Fallback is kept for older systems.
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                dtype=np.float32,
            )
        except TypeError:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=True,
                dtype=np.float32,
            )

    def _build_preprocessor(self, num, cat):
        transformers = []

        if num:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="median")),
                            (
                                "sc",
                                StandardScaler(
                                    with_mean=False,
                                    copy=False,
                                ),
                            ),
                        ]
                    ),
                    num,
                )
            )

        if cat:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            (
                                "imp",
                                SimpleImputer(
                                    strategy="most_frequent",
                                ),
                            ),
                            ("oh", self._make_encoder()),
                        ]
                    ),
                    cat,
                )
            )

        return ColumnTransformer(
            transformers,
            remainder="drop",
            sparse_threshold=0.30,
        )

    @staticmethod
    def _dense_if_needed(x):
        """Convert sparse matrices to dense only for algorithms that require it."""
        if hasattr(x, "toarray"):
            return x.toarray().astype(np.float32, copy=False)
        return np.asarray(x, dtype=np.float32)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _make_models(self, n_classes):
        models = {
            "Logistic Regression": LogisticRegression(
                C=0.25,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1500,
                tol=1e-4,
                random_state=RANDOM_STATE,
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=450,
                max_depth=10,
                min_samples_split=30,
                min_samples_leaf=12,
                max_features="sqrt",
                class_weight="balanced_subsample",
                bootstrap=True,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),

            "Extra Trees": ExtraTreesClassifier(
                n_estimators=450,
                max_depth=14,
                min_samples_split=25,
                min_samples_leaf=8,
                max_features="sqrt",
                class_weight="balanced",
                bootstrap=False,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),

            "ANN": MLPClassifier(
                hidden_layer_sizes=(96, 48, 24),
                activation="relu",
                solver="adam",
                alpha=0.01,
                batch_size=512,
                learning_rate="adaptive",
                learning_rate_init=0.0003,
                max_iter=120,
                early_stopping=True,
                validation_fraction=0.10,
                n_iter_no_change=10,
                tol=1e-4,
                max_fun=15000,
                random_state=RANDOM_STATE,
            ),

            "XGBoost": XGBClassifier(
                n_estimators=1500,
                max_depth=4,
                learning_rate=0.025,
                min_child_weight=20,
                gamma=0.10,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=8.0,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                tree_method="hist",
                early_stopping_rounds=60,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

            "HistGradientBoosting": HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=350,
                max_leaf_nodes=20,
                max_depth=6,
                min_samples_leaf=120,
                l2_regularization=8.0,
                max_bins=255,
                early_stopping=True,
                validation_fraction=0.10,
                n_iter_no_change=20,
                tol=1e-5,
                random_state=RANDOM_STATE,
            ),
        }

        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
                n_estimators=1500,
                learning_rate=0.03,
                num_leaves=20,
                max_depth=6,
                min_child_samples=150,
                min_split_gain=0.05,
                subsample=0.80,
                subsample_freq=1,
                colsample_bytree=0.80,
                reg_alpha=0.5,
                reg_lambda=8.0,
                max_bin=255,
                verbosity=-1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            self.log("LightGBM not installed -> model will be skipped.")

        if CATBOOST_AVAILABLE:
            models["CatBoost"] = CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="MultiClass",
                iterations=1500,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=10.0,
                random_strength=1.0,
                border_count=128,
                random_seed=RANDOM_STATE,
                verbose=False,
                thread_count=-1,
                allow_writing_files=False,
            )
        else:
            self.log("CatBoost not installed -> model will be skipped.")

        return models

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metric_dict(
        name,
        y_train,
        train_pred,
        y_val,
        val_pred,
        y_test=None,
        test_pred=None,
    ):
        train_f1 = f1_score(
            y_train,
            train_pred,
            average="macro",
        )
        val_f1 = f1_score(
            y_val,
            val_pred,
            average="macro",
        )
        gap = max(0.0, train_f1 - val_f1)

        d = {
            "Model": name,
            "Train Accuracy": accuracy_score(y_train, train_pred),
            "Train Macro F1": train_f1,
            "Validation Accuracy": accuracy_score(y_val, val_pred),
            "Validation Balanced Accuracy": balanced_accuracy_score(y_val, val_pred),
            "Validation Macro F1": val_f1,
            "Validation Weighted F1": f1_score(
                y_val,
                val_pred,
                average="weighted",
            ),
            "Overfit Gap": gap,
            "Overfit Status": (
                "HIGH"
                if gap >= OVERFIT_GAP_HARD
                else "WARNING"
                if gap >= OVERFIT_GAP_WARN
                else "OK"
            ),
        }

        if y_test is not None and test_pred is not None:
            d.update(
                {
                    "Test Accuracy": accuracy_score(y_test, test_pred),
                    "Test Balanced Accuracy": balanced_accuracy_score(
                        y_test,
                        test_pred,
                    ),
                    "Test Macro F1": f1_score(
                        y_test,
                        test_pred,
                        average="macro",
                    ),
                    "Test Weighted F1": f1_score(
                        y_test,
                        test_pred,
                        average="weighted",
                    ),
                }
            )

        return d

    def _fit_one(
        self,
        name,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        train_weight,
        val_weight,
    ):
        self.log("\n" + "=" * 60)
        self.log("TRAINING: " + name)
        self.log("=" * 60)

        model_start = time.perf_counter()
        self.log(f"⏱ START TIME: {time.strftime('%H:%M:%S')}")

        try:
            if name == "XGBoost":
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=train_weight,
                    eval_set=[(X_val, y_val)],
                    sample_weight_eval_set=[val_weight],
                    verbose=False,
                )

            elif name == "LightGBM":
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=train_weight,
                    eval_set=[(X_val, y_val)],
                    eval_sample_weight=[val_weight],
                    callbacks=[
                        lgb_early_stopping(
                            stopping_rounds=60,
                            verbose=False,
                        )
                    ],
                )

            elif name == "CatBoost":
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=train_weight,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=60,
                    verbose=False,
                )

            elif name == "ANN":
                # MLPClassifier performs its own internal validation-based early stopping.
                # sample_weight is supported in recent scikit-learn versions.
                try:
                    model.fit(
                        X_train,
                        y_train,
                        sample_weight=train_weight,
                    )
                except TypeError:
                    self.log(
                        "ANN sample_weight unsupported in this sklearn build; "
                        "training without explicit sample weights."
                    )
                    model.fit(X_train, y_train)

            else:
                try:
                    model.fit(
                        X_train,
                        y_train,
                        sample_weight=train_weight,
                    )
                except TypeError:
                    model.fit(X_train, y_train)

        except Exception:
            elapsed = time.perf_counter() - model_start
            self.log(
                f"⏱ {name} FAILED after "
                f"{elapsed:.2f}s ({elapsed/60:.2f} min)"
            )
            raise

        elapsed = time.perf_counter() - model_start
        try:
            model._aegis_training_seconds = elapsed
        except Exception:
            pass
        self.log(f"⏱ END TIME: {time.strftime('%H:%M:%S')}")
        self.log(
            f"✓ {name} TRAINING COMPLETED IN "
            f"{elapsed:.2f} seconds ({elapsed/60:.2f} minutes)"
        )
        self.log("-" * 60)

        return model

    # ------------------------------------------------------------------
    # Main training pipeline
    # ------------------------------------------------------------------

    def _set_training_buttons_state(self, enabled):
        state = "normal" if enabled else "disabled"
        for button in getattr(self, "individual_train_buttons", {}).values():
            button.config(state=state)

    def start(self):
        """Train all available models."""
        if self.training:
            return

        f = self.dataset.get()
        if not os.path.exists(f):
            messagebox.showerror(
                "Dataset Not Found",
                "Select a valid CSV dataset.",
            )
            return

        self.training = True
        self.training_session_started_at = time.perf_counter()
        self.status.set("Training all models...")
        self.pb.start(12)
        self.animation_phase = 0
        self.animate_training()
        self._set_training_buttons_state(False)
        self.nb.select(self.con)

        threading.Thread(
            target=self.train,
            args=(f, None),
            daemon=True,
        ).start()

    def start_single(self, model_name):
        """Train exactly one selected model and make it the saved production model."""
        if self.training:
            return

        f = self.dataset.get()
        if not os.path.exists(f):
            messagebox.showerror(
                "Dataset Not Found",
                "Select a valid CSV dataset.",
            )
            return

        self.training = True
        self.training_session_started_at = time.perf_counter()
        self.status.set(f"Training {model_name}...")
        self.pb.start(12)
        self.animation_phase = 0
        self.animate_training()
        self._set_training_buttons_state(False)
        self.nb.select(self.con)

        self.log("")
        self.log("=" * 72)
        self.log(f"INDIVIDUAL MODEL TRAINING REQUESTED: {model_name}")
        self.log("Only this model will be trained.")
        self.log("=" * 72)

        threading.Thread(
            target=self.train,
            args=(f, model_name),
            daemon=True,
        ).start()

    def train(self, path, only_model=None):
        """
        Train either all models or exactly one model.

        only_model=None  -> Train All Models + validation-based selection.
        only_model=name   -> Train only that model and make it production model.

        The test set is never used for model selection.
        """
        try:
            self.log("=" * 72)
            self.log("AEGISAI LARGE-TABULAR MODEL TRAINING STARTED")
            self.log("=" * 72)

            if only_model:
                self.log(f"MODE: INDIVIDUAL TRAINING -> {only_model}")
            else:
                self.log("MODE: TRAIN ALL MODELS")

            df = pd.read_csv(path)
            self.log(f"Original dataset shape: {df.shape}")
            self.log(f"Missing cells: {int(df.isnull().sum().sum()):,}")

            if TARGET not in df.columns:
                raise ValueError(
                    f"Target column '{TARGET}' was not found in the dataset."
                )

            if len(df) > MAX_DATASET_ROWS:
                self.log(
                    f"Dataset exceeds {MAX_DATASET_ROWS:,} rows. "
                    f"Using a stratified {MAX_DATASET_ROWS:,}-row sample."
                )
                y_all = df[TARGET].copy()
                X_all = df.drop(
                    columns=[c for c in DROP_COLUMNS if c in df.columns],
                    errors="ignore",
                )
                X, y = self._safe_stratified_cap(
                    X_all, y_all, MAX_DATASET_ROWS
                )
                df = X.copy()
                df[TARGET] = y
            else:
                y = df[TARGET].copy()
                X = df.drop(
                    columns=[c for c in DROP_COLUMNS if c in df.columns],
                    errors="ignore",
                )

            all_null = X.columns[X.isnull().all()].tolist()
            if all_null:
                self.log(
                    "Dropping all-null columns: "
                    + ", ".join(map(str, all_null))
                )
                X = X.drop(columns=all_null)

            constant = [
                c for c in X.columns
                if X[c].nunique(dropna=False) <= 1
            ]
            if constant:
                self.log(
                    "Dropping constant columns: "
                    + ", ".join(map(str, constant))
                )
                X = X.drop(columns=constant)

            self.log("Target distribution:")
            for k, v in y.value_counts(dropna=False).items():
                self.log(f"  {k}: {v:,}")

            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=TEST_SIZE + 0.15,
                random_state=RANDOM_STATE,
                stratify=y,
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=0.50,
                random_state=RANDOM_STATE,
                stratify=y_temp,
            )

            self.log(
                f"Split sizes: train={len(X_train):,} | "
                f"validation={len(X_val):,} | "
                f"test={len(X_test):,}"
            )

            cat = X_train.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()
            num = [c for c in X_train.columns if c not in cat]

            self.log(
                f"Features: {X_train.shape[1]} | "
                f"Numerical: {len(num)} | Categorical: {len(cat)}"
            )

            pre = self._build_preprocessor(num, cat)
            self.log("Fitting leakage-safe preprocessing on TRAIN only...")
            X_train_p = pre.fit_transform(X_train)
            X_val_p = pre.transform(X_val)
            X_test_p = pre.transform(X_test)

            self.log(
                f"Processed shapes: train={X_train_p.shape}, "
                f"val={X_val_p.shape}, test={X_test_p.shape}"
            )

            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_val_enc = le.transform(y_val)
            y_test_enc = le.transform(y_test)
            n_classes = len(le.classes_)

            if n_classes < 2:
                raise ValueError("The target must contain at least two classes.")

            train_weight = compute_sample_weight(
                class_weight="balanced", y=y_train_enc
            )
            val_weight = compute_sample_weight(
                class_weight="balanced", y=y_val_enc
            )

            self.log(
                f"Number of target classes: {n_classes} "
                f"({', '.join(map(str, le.classes_))})"
            )

            X_train_dense = None
            X_val_dense = None
            X_test_dense = None

            models = self._make_models(n_classes)

            if only_model:
                if only_model not in models:
                    if only_model == "LightGBM":
                        raise RuntimeError(
                            "LightGBM is not installed. Install it with: pip install lightgbm"
                        )
                    if only_model == "CatBoost":
                        raise RuntimeError(
                            "CatBoost is not installed. Install it with: pip install catboost"
                        )
                    raise ValueError(f"Unknown model: {only_model}")
                models = {only_model: models[only_model]}

            results = []
            trained = {}
            model_predictions = {}
            model_metrics = {}

            for name, model in models.items():
                try:
                    if name == "HistGradientBoosting":
                        if X_train_dense is None:
                            self.log(
                                "Converting processed matrix to dense for "
                                "HistGradientBoosting..."
                            )
                            X_train_dense = self._dense_if_needed(X_train_p)
                            X_val_dense = self._dense_if_needed(X_val_p)
                            X_test_dense = self._dense_if_needed(X_test_p)
                        fit_train = X_train_dense
                        fit_val = X_val_dense
                    else:
                        fit_train = X_train_p
                        fit_val = X_val_p

                    model = self._fit_one(
                        name,
                        model,
                        fit_train,
                        y_train_enc,
                        fit_val,
                        y_val_enc,
                        train_weight,
                        val_weight,
                    )

                    train_pred = model.predict(fit_train)
                    val_pred = model.predict(fit_val)

                    d = self._metric_dict(
                        name,
                        y_train_enc, train_pred,
                        y_val_enc, val_pred,
                    )

                    results.append(d)
                    trained[name] = model
                    model_predictions[name] = {
                        "train": train_pred,
                        "validation": val_pred,
                    }
                    model_metrics[name] = d

                    self.log(
                        f"Train Macro F1: {d['Train Macro F1']:.4f} | "
                        f"Val Macro F1: {d['Validation Macro F1']:.4f} | "
                        f"Gap: {d['Overfit Gap']:.4f} | "
                        f"Status: {d['Overfit Status']}"
                    )
                    self.log(
                        f"MODEL FINISHED: {name} | "
                        f"Training duration: "
                        f"{getattr(model, '_aegis_training_seconds', 0.0):.2f}s"
                    )
                except Exception as model_error:
                    self.log(f"{name} FAILED: {model_error}")
                    self.log(traceback.format_exc())

            if not results:
                raise RuntimeError(
                    "All requested model trainings failed. Check the console."
                )

            rdf = pd.DataFrame(results)

            rdf["Selection Score"] = (
                rdf["Validation Macro F1"]
                - 0.50
                * np.maximum(
                    rdf["Overfit Gap"] - OVERFIT_GAP_WARN,
                    0.0,
                )
            )

            if only_model:
                selected_name = only_model
                selected_model = trained[selected_name]
                requested_gap = float(rdf.loc[rdf["Model"] == selected_name, "Overfit Gap"].iloc[0])
                self.log("\nMODEL SELECTION BY USER REQUEST")
                self.log(f"Selected model: {selected_name}")
                if requested_gap >= OVERFIT_GAP_HARD:
                    self.log(
                        f"WARNING: {selected_name} has an overfit gap of {requested_gap:.2%}. "
                        "It will be evaluated, but will NOT be promoted as the production model."
                    )
            else:
                acceptable = rdf[
                    rdf["Overfit Gap"] < OVERFIT_GAP_HARD
                ].copy()
                selection_pool = (
                    acceptable if not acceptable.empty else rdf.copy()
                )
                selection_pool = selection_pool.sort_values(
                    ["Selection Score", "Validation Macro F1"],
                    ascending=False,
                ).reset_index(drop=True)

                selected_name = str(selection_pool.iloc[0]["Model"])
                selected_model = trained[selected_name]

                self.log("\n" + "=" * 72)
                self.log("MODEL SELECTION — TEST SET STILL UNTOUCHED")
                self.log("=" * 72)
                self.log(
                    f"Selected: {selected_name}\n"
                    f"Validation Macro F1: "
                    f"{selection_pool.iloc[0]['Validation Macro F1']:.4f}\n"
                    f"Train→Validation gap: "
                    f"{selection_pool.iloc[0]['Overfit Gap']:.4f}"
                )

            if selected_name == "HistGradientBoosting":
                if X_test_dense is None:
                    X_train_dense = self._dense_if_needed(X_train_p)
                    X_val_dense = self._dense_if_needed(X_val_p)
                    X_test_dense = self._dense_if_needed(X_test_p)
                selected_test_input = X_test_dense
            else:
                selected_test_input = X_test_p

            test_pred = selected_model.predict(selected_test_input)

            final_metrics = self._metric_dict(
                selected_name,
                y_train_enc,
                model_predictions[selected_name]["train"],
                y_val_enc,
                model_predictions[selected_name]["validation"],
                y_test_enc,
                test_pred,
            )

            # ----------------------------------------------------------
            # FINAL TEST EVALUATION FOR EVERY TRAINED MODEL
            # ----------------------------------------------------------
            # Model selection is already finished using ONLY validation
            # Macro F1 + overfitting gap. Now that the production candidate
            # has been selected, evaluate every trained candidate on the
            # untouched test set so the dashboard can show Test F1/Accuracy
            # for every model instead of only the selected model.
            # Test metrics are NEVER used to choose the production model.
            test_columns = [
                "Test Accuracy",
                "Test Balanced Accuracy",
                "Test Macro F1",
                "Test Weighted F1",
            ]

            for c in test_columns:
                if c not in rdf.columns:
                    rdf[c] = np.nan

            all_test_metrics = {}
            for name, candidate_model in trained.items():
                if name == "HistGradientBoosting":
                    candidate_test_input = X_test_dense
                    if candidate_test_input is None:
                        candidate_test_input = self._dense_if_needed(X_test_p)
                else:
                    candidate_test_input = X_test_p

                candidate_test_pred = (
                    test_pred
                    if name == selected_name
                    else candidate_model.predict(candidate_test_input)
                )

                candidate_metrics = self._metric_dict(
                    name,
                    y_train_enc,
                    model_predictions[name]["train"],
                    y_val_enc,
                    model_predictions[name]["validation"],
                    y_test_enc,
                    candidate_test_pred,
                )

                model_predictions[name]["test"] = candidate_test_pred
                model_metrics[name] = candidate_metrics
                all_test_metrics[name] = candidate_metrics

                for col in test_columns:
                    rdf.loc[rdf["Model"] == name, col] = candidate_metrics[col]

                self.log(
                    f"TEST EVALUATION: {name} | "
                    f"Accuracy={candidate_metrics['Test Accuracy']:.4f} | "
                    f"Macro F1={candidate_metrics['Test Macro F1']:.4f}"
                )

            # Keep the selected model's metrics as the canonical final
            # production metrics. Its values are identical to the loop above.
            final_metrics = all_test_metrics[selected_name]

            rdf = rdf.sort_values(
                "Validation Macro F1", ascending=False
            ).reset_index(drop=True)

            self.log("\n" + "=" * 72)
            self.log("FINAL TEST EVALUATION — ALL TRAINED MODELS")
            self.log("=" * 72)
            self.log(f"Production/selected model: {selected_name}")
            self.log(
                f"Test Accuracy: {final_metrics['Test Accuracy']:.4f}"
            )
            self.log(
                f"Test Balanced Accuracy: "
                f"{final_metrics['Test Balanced Accuracy']:.4f}"
            )
            self.log(
                f"Test Macro F1: {final_metrics['Test Macro F1']:.4f}"
            )
            self.log(
                f"Test Weighted F1: "
                f"{final_metrics['Test Weighted F1']:.4f}"
            )

            self.log("\n" + "=" * 72)
            self.log("MODEL COMPARISON")
            self.log("=" * 72)

            display_cols = [
                "Model",
                "Train Macro F1",
                "Validation Macro F1",
                "Test Macro F1",
                "Overfit Gap",
                "Overfit Status",
            ]
            self.log(rdf[display_cols].to_string(index=False))

            self.log("\nCLASSIFICATION REPORT — SELECTED PRODUCTION MODEL")
            self.log(
                classification_report(
                    y_test_enc,
                    test_pred,
                    target_names=le.classes_,
                    digits=4,
                )
            )

            self.log("CONFUSION MATRIX — SELECTED PRODUCTION MODEL")
            for row in confusion_matrix(y_test_enc, test_pred):
                self.log("  " + "  ".join(map(str, row)))

            gap = final_metrics["Overfit Gap"]
            if gap <= OVERFIT_GAP_GOOD:
                self.log(
                    f"\nOVERFITTING CHECK: EXCELLENT — gap <= "
                    f"{OVERFIT_GAP_GOOD:.2%}."
                )
            elif gap <= OVERFIT_GAP_WARN:
                self.log(
                    f"\nOVERFITTING CHECK: GOOD — gap <= "
                    f"{OVERFIT_GAP_WARN:.2%}."
                )
            elif gap >= OVERFIT_GAP_HARD:
                self.log(
                    "\nOVERFITTING WARNING: selected model has a large "
                    "train→validation Macro-F1 gap."
                )
            else:
                self.log(
                    "\nOVERFITTING NOTICE: selected model has a moderate "
                    "train→validation Macro-F1 gap."
                )

            # Store the final test metrics for the currently trained model
            # before merging it into the cumulative registry.
            self.model_metrics[selected_name] = dict(final_metrics)

            # Keep the test prediction alongside its train/validation
            # predictions for this model.
            model_predictions[selected_name]["test"] = test_pred

            # ----------------------------------------------------------
            # CUMULATIVE MODEL REGISTRY / BEST-MODEL DECISION
            # ----------------------------------------------------------
            # Individual training must NOT erase models trained earlier.
            # Every trained model remains in memory and its metrics remain
            # visible in the dashboard. Only the best production model is
            # written to MODEL_PATH.
            dataset_key = os.path.abspath(path)

            # If the user switches to a different dataset, the old models
            # are not comparable and therefore start a fresh registry.
            if (
                self.training_dataset_path is not None
                and os.path.abspath(self.training_dataset_path) != dataset_key
            ):
                self.log(
                    "\nDATASET CHANGED -> resetting cumulative model registry."
                )
                self.trained_models.clear()
                self.model_predictions.clear()
                self.model_metrics.clear()
                self.model_preprocessors.clear()
                self.model_label_encoders.clear()
                self.model_features.clear()
                self.results_df = None
                self.best_model = None
                self.best_model_name = None
                self.best_model_metrics = None

            self.training_dataset_path = dataset_key

            # Merge newly trained models into the existing registry.
            self.trained_models.update(trained)
            self.model_predictions.update(model_predictions)
            self.model_metrics.update(model_metrics)

            # Keep preprocessing/label information for every model.
            for model_name in trained:
                self.model_preprocessors[model_name] = pre
                self.model_label_encoders[model_name] = le
                self.model_features[model_name] = X.columns.tolist()

            # Add/replace the newly evaluated model rows without deleting
            # earlier individual-training results.
            cumulative_rows = []
            for model_name, metrics in self.model_metrics.items():
                row = dict(metrics)
                row["Selection Score"] = (
                    row["Validation Macro F1"]
                    - 0.50
                    * max(
                        row["Overfit Gap"] - OVERFIT_GAP_WARN,
                        0.0,
                    )
                )
                cumulative_rows.append(row)

            cumulative_df = pd.DataFrame(cumulative_rows)

            # The current candidate is already fully test-evaluated.
            candidate_metrics = dict(final_metrics)
            candidate_metrics["Selection Score"] = (
                candidate_metrics["Validation Macro F1"]
                - 0.50
                * max(
                    candidate_metrics["Overfit Gap"] - OVERFIT_GAP_WARN,
                    0.0,
                )
            )
            candidate_metrics["Production Eligible"] = (
                candidate_metrics["Overfit Gap"] < OVERFIT_GAP_HARD
            )

            previous_best_name = self.best_model_name
            previous_best_metrics = (
                dict(self.best_model_metrics)
                if self.best_model_metrics is not None
                else None
            )

            def quality_score(metrics):
                gap = metrics.get("Overfit Gap", 0.0)
                if gap >= OVERFIT_GAP_HARD:
                    return -np.inf
                return (
                    metrics.get("Validation Macro F1", -np.inf)
                    - 0.50
                    * max(gap - OVERFIT_GAP_WARN, 0.0)
                )

            candidate_score = quality_score(candidate_metrics)

            # First model becomes the production model.
            # Later models replace it ONLY when they are better by the same
            # validation/overfit criterion used by Train All Models.
            replace_best = (
                previous_best_metrics is None
                or candidate_score > quality_score(previous_best_metrics)
            )

            if replace_best:
                self.best_model = trained[selected_name]
                self.best_model_name = selected_name
                self.best_model_metrics = final_metrics
                self.preprocessor = pre
                self.label_encoder = le
                self.features = X.columns.tolist()
                self.num_features = num
                self.cat_features = cat

                self.log("\n" + "=" * 72)
                self.log("PRODUCTION MODEL UPDATE")
                self.log("=" * 72)
                if previous_best_name is None:
                    self.log(
                        f"FIRST MODEL -> {selected_name} is now the production model."
                    )
                else:
                    self.log(
                        f"NEW BEST -> {selected_name} replaced "
                        f"{previous_best_name}."
                    )
                    self.log(
                        f"Previous selection score: "
                        f"{quality_score(previous_best_metrics):.6f}"
                    )
                self.log(
                    f"New selection score: {candidate_score:.6f}"
                )
                self.log(
                    f"Validation Macro F1: "
                    f"{final_metrics['Validation Macro F1']:.4f}"
                )
                self.log(
                    f"Test Macro F1: "
                    f"{final_metrics['Test Macro F1']:.4f}"
                )
                self.log(
                    "ACTION: Saving this model as the production/best model."
                )

                # Only the best model is persisted to the default production
                # file. Older trained models stay in self.trained_models.
                self.save(silent=True)
                save_message = (
                    f"BEST MODEL SAVED: {self.best_model_name} "
                    f"-> {os.path.abspath(MODEL_PATH)}"
                )
                self.log(save_message)
            else:
                # IMPORTANT: Do not touch self.best_model or MODEL_PATH.
                self.log("\n" + "=" * 72)
                self.log("PRODUCTION MODEL KEPT")
                self.log("=" * 72)
                self.log(
                    f"Current best: {previous_best_name}"
                )
                self.log(
                    f"Current best selection score: "
                    f"{quality_score(previous_best_metrics):.6f}"
                )
                self.log(
                    f"New candidate: {selected_name}"
                )
                self.log(
                    f"New candidate selection score: {candidate_score:.6f}"
                )
                self.log(
                    "ACTION: Candidate retained in memory, "
                    "but NOT saved as production model."
                )
                self.log(
                    f"Production file remains: {os.path.abspath(MODEL_PATH)}"
                )

            # Build the cumulative dashboard table.
            if not cumulative_df.empty:
                cumulative_df = cumulative_df.sort_values(
                    ["Selection Score", "Validation Macro F1"],
                    ascending=False,
                ).reset_index(drop=True)

            self.df = df
            self.results_df = cumulative_df

            # Keep the report focused on the actual production model.
            mode_text = (
                f"INDIVIDUAL MODEL: {selected_name}"
                if only_model else "ALL MODELS"
            )

            best_for_report = self.best_model_metrics

            self.report_text = (
                "AEGISAI MODEL EVALUATION REPORT\n"
                + "=" * 76
                + f"\n\nTRAINING MODE: {mode_text}\n"
                "TRAINING ARCHITECTURE\n"
                "• Train/Validation/Test = 70/15/15\n"
                "• Preprocessor fit on TRAIN only\n"
                "• Test set remains untouched until final evaluation\n"
                "• Strong regularization + early stopping where supported\n"
                "• Overfit gap = Train Macro F1 - Validation Macro F1\n"
                "• Preferred overfit gap = 0–2%\n"
                "• Individual training is cumulative; previous models are retained\n"
                "• Only the best production model is saved to the default model path\n"
                "\nCUMULATIVE MODEL COMPARISON\n"
                + cumulative_df.to_string(index=False)
                + "\n\nPRODUCTION MODEL: " + str(self.best_model_name)
                + "\n\nPRODUCTION MODEL TEST METRICS\n"
                + "\n".join([
                    f"Accuracy: {best_for_report['Test Accuracy']:.4f}",
                    f"Balanced Accuracy: "
                    f"{best_for_report['Test Balanced Accuracy']:.4f}",
                    f"Macro F1: {best_for_report['Test Macro F1']:.4f}",
                    f"Weighted F1: "
                    f"{best_for_report['Test Weighted F1']:.4f}",
                    f"Train Macro F1: "
                    f"{best_for_report['Train Macro F1']:.4f}",
                    f"Validation Macro F1: "
                    f"{best_for_report['Validation Macro F1']:.4f}",
                    f"Overfit Gap: "
                    f"{best_for_report['Overfit Gap']:.4f}",
                    f"Overfit Status: "
                    f"{best_for_report['Overfit Status']}",
                ])
                + "\n"
            )

            self.q.put(("done", None))

        except Exception as e:
            self.log(
                "\nTRAINING ERROR\n" + traceback.format_exc()
            )
            self.q.put(("error", str(e)))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def bundle(self):
        return {
            "model": self.best_model,
            "best_model_name": self.best_model_name,
            "best_model_metrics": self.best_model_metrics,
            "preprocessor": self.preprocessor,
            "label_encoder": self.label_encoder,
            "features": self.features,
            "numeric_features": self.num_features,
            "categorical_features": self.cat_features,
            "target": TARGET,
            "excluded_columns": DROP_COLUMNS,
            "architecture_version": "AegisAI-300K-v3-IndividualTraining",
            "random_state": RANDOM_STATE,
            "train_validation_test_split": "70/15/15",
            "overfit_gap_good": OVERFIT_GAP_GOOD,
            "overfit_gap_warn": OVERFIT_GAP_WARN,
            "overfit_gap_hard": OVERFIT_GAP_HARD,
        }

    def save(self, silent=False):
        if self.best_model is None:
            if not silent:
                messagebox.showwarning(
                    "No Model",
                    "Train or load a model first.",
                )
            return

        try:
            joblib.dump(
                self.bundle(),
                MODEL_PATH,
                compress=3,
            )

            self.log(
                "MODEL SAVED: "
                + os.path.abspath(MODEL_PATH)
            )

            if not silent:
                messagebox.showinfo(
                    "Saved",
                    (
                        "Best model saved successfully.\n\n"
                        + os.path.abspath(MODEL_PATH)
                    ),
                )

        except Exception as e:
            if not silent:
                messagebox.showerror(
                    "Save Error",
                    str(e),
                )

    # ------------------------------------------------------------------
    # Animation / external GUI
    # ------------------------------------------------------------------

    def animate_training(self):
        if not self.training:
            return

        dots = [
            "●",
            "● ●",
            "● ● ●",
            "● ● ● ●",
        ]
        self.animation_phase = (
            self.animation_phase + 1
        ) % len(dots)

        self.anim_label.config(
            text="TRAINING " + dots[self.animation_phase],
            fg=ACCENT,
        )

        self.after(
            350,
            self.animate_training,
        )
    def open_v3_file(self):
        """Run V3_AegisAI_GUI_Adaptive_Multimodal_Updated.py in a new process."""
        path = Path.cwd() / SOURCE_GUI_FILE
        if not path.exists():
            path = (
                Path(__file__).resolve().parent
                / SOURCE_GUI_FILE
            )

        if not path.exists():
            messagebox.showerror(
                "V3 GUI Not Found",
                (
                    f"Could not find:\n\n"
                    f"{SOURCE_GUI_FILE}\n\n"
                    "Place it in the same folder as this GUI."
                ),
            )
            return

        try:
            if os.name == "nt":
                creationflags = getattr(
                    subprocess,
                    "CREATE_NEW_PROCESS_GROUP",
                    0,
                )
                subprocess.Popen(
                    [sys.executable, str(path)],
                    cwd=str(path.parent),
                    creationflags=creationflags,
                )
            else:
                subprocess.Popen(
                    [sys.executable, str(path)],
                    cwd=str(path.parent),
                )

            self.log(
                f"V3 GUI LAUNCHED: {path}"
            )
            self.status.set("V3 GUI launched")

        except Exception as e:
            self.log(
                f"V3 LAUNCH ERROR: {e}"
            )
            messagebox.showerror(
                "V3 GUI Launch Error",
                (
                    f"Could not run {SOURCE_GUI_FILE}.\n\n"
                    f"{e}"
                ),
            )

    # ------------------------------------------------------------------
    # Report / CSV
    # ------------------------------------------------------------------

    def save_report(self):
        if not self.report_text:
            messagebox.showwarning(
                "No Report",
                "Train the models first so the report is available.",
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save AegisAI Training Report",
            defaultextension=".txt",
            filetypes=[
                ("Text report", "*.txt"),
                ("Markdown report", "*.md"),
                ("All files", "*.*"),
            ],
            initialfile="AegisAI_Model_Evaluation_Report.txt",
        )

        if not path:
            return

        try:
            Path(path).write_text(
                self.report_text,
                encoding="utf-8",
            )

            self.log(
                f"REPORT SAVED: {path}"
            )

            messagebox.showinfo(
                "Report Saved",
                f"Report saved successfully:\n\n{path}",
            )

        except Exception as e:
            messagebox.showerror(
                "Report Error",
                str(e),
            )

    def export_csv(self):
        if self.results_df is None:
            messagebox.showwarning(
                "No Metrics",
                "Train the models first so model metrics are available.",
            )
            return

        path = filedialog.asksaveasfilename(
            title="Export Model Metrics CSV",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
            initialfile="AegisAI_Model_Metrics.csv",
        )

        if not path:
            return

        try:
            export_df = self.results_df.copy()

            export_df["Is_Selected_Model"] = (
                export_df["Model"]
                == self.best_model_name
            )

            for col in [
                "Train Macro F1",
                "Validation Macro F1",
                "Test Macro F1",
                "Overfit Gap",
            ]:
                if col in export_df.columns:
                    export_df[col + " Percent"] = (
                        export_df[col] * 100
                    )

            export_df.to_csv(
                path,
                index=False,
            )

            self.log(
                f"METRICS CSV SAVED: {path}"
            )

            messagebox.showinfo(
                "CSV Saved",
                (
                    "Model metrics exported successfully.\n\n"
                    + path
                ),
            )

        except Exception as e:
            messagebox.showerror(
                "CSV Export Error",
                str(e),
            )

    # ------------------------------------------------------------------
    # Individual model saving
    # ------------------------------------------------------------------

    def save_individual_model(self):
        if not self.trained_models:
            messagebox.showwarning(
                "No Models",
                "Train all models first.",
            )
            return

        win = tk.Toplevel(self)
        win.title("Save Individual AegisAI Model")
        win.geometry("520x360")
        win.configure(bg=PANEL)
        win.transient(self)
        win.grab_set()

        tk.Label(
            win,
            text="Save Individual Model",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 15),
        ).pack(
            anchor="w",
            padx=24,
            pady=(22, 5),
        )

        tk.Label(
            win,
            text=(
                "Choose a trained model, rename it, "
                "and select any location."
            ),
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(
            anchor="w",
            padx=24,
            pady=(0, 18),
        )

        tk.Label(
            win,
            text="Model",
            bg=PANEL,
            fg=MUTED,
        ).pack(anchor="w", padx=24)

        model_var = tk.StringVar(
            value=list(self.trained_models.keys())[0]
        )

        combo = ttk.Combobox(
            win,
            textvariable=model_var,
            values=list(self.trained_models.keys()),
            state="readonly",
            width=48,
        )
        combo.pack(
            fill="x",
            padx=24,
            pady=(5, 15),
        )

        tk.Label(
            win,
            text="Custom file name",
            bg=PANEL,
            fg=MUTED,
        ).pack(anchor="w", padx=24)

        default_name = (
            self.best_model_name.replace(" ", "_")
            + "_AegisAI.joblib"
        )

        name_var = tk.StringVar(
            value=default_name
        )

        tk.Entry(
            win,
            textvariable=name_var,
            bg="#0A1524",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Segoe UI", 10),
        ).pack(
            fill="x",
            padx=24,
            pady=(5, 15),
        )

        def do_save():
            selected = model_var.get()
            filename = name_var.get().strip()

            if not filename:
                messagebox.showerror(
                    "Invalid Name",
                    "Enter a model file name.",
                    parent=win,
                )
                return

            if not filename.lower().endswith(".joblib"):
                filename += ".joblib"

            path = filedialog.asksaveasfilename(
                parent=win,
                title=f"Save {selected}",
                defaultextension=".joblib",
                initialfile=filename,
                filetypes=[
                    ("AegisAI Model", "*.joblib"),
                    ("All files", "*.*"),
                ],
            )

            if not path:
                return

            try:
                metrics = self.model_metrics.get(
                    selected,
                    {},
                )

                bundle = {
                    "model": self.trained_models[selected],
                    "best_model_name": selected,
                    "best_model_metrics": metrics,
                    "preprocessor": self.preprocessor,
                    "label_encoder": self.label_encoder,
                    "features": self.features,
                    "numeric_features": self.num_features,
                    "categorical_features": self.cat_features,
                    "target": TARGET,
                    "excluded_columns": DROP_COLUMNS,
                    "model_type": "individual_trained_model",
                    "architecture_version": "AegisAI-300K-v3-IndividualTraining",
                }

                joblib.dump(
                    bundle,
                    path,
                    compress=3,
                )

                self.log(
                    f"INDIVIDUAL MODEL SAVED: "
                    f"{selected} -> {path}"
                )

                messagebox.showinfo(
                    "Model Saved",
                    (
                        f"{selected} saved successfully.\n\n"
                        f"{path}"
                    ),
                    parent=win,
                )

                win.destroy()

            except Exception as e:
                messagebox.showerror(
                    "Save Error",
                    str(e),
                    parent=win,
                )

        self.btn(
            win,
            "💾  Save Model",
            do_save,
            ACCENT,
        ).pack(pady=18)

    # ------------------------------------------------------------------
    # Load / delete / dashboard
    # ------------------------------------------------------------------

    def load(self):
        f = filedialog.askopenfilename(
            filetypes=[
                ("Joblib model", "*.joblib"),
                ("All files", "*.*"),
            ]
        )

        if not f:
            return

        try:
            b = joblib.load(f)

            self.best_model = b["model"]
            self.best_model_name = b["best_model_name"]
            self.best_model_metrics = b["best_model_metrics"]

            self.preprocessor = b["preprocessor"]
            self.label_encoder = b["label_encoder"]
            self.features = b["features"]

            self.num_features = b.get(
                "numeric_features",
                [],
            )
            self.cat_features = b.get(
                "categorical_features",
                [],
            )

            self.update_dash_loaded()

            self.status.set("Model loaded")

            self.log(
                f"MODEL LOADED: {f}"
            )

        except Exception as e:
            messagebox.showerror(
                "Load Error",
                str(e),
            )

    def delete(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showinfo(
                "No Model",
                "No default saved model exists.",
            )
            return

        if messagebox.askyesno(
            "Delete Saved Model",
            (
                f"Delete:\n"
                f"{os.path.abspath(MODEL_PATH)}?"
            ),
        ):
            try:
                os.remove(MODEL_PATH)

                self.log(
                    "MODEL DELETED: "
                    + os.path.abspath(MODEL_PATH)
                )

                self.status.set(
                    "Saved model deleted"
                )

            except Exception as e:
                messagebox.showerror(
                    "Delete Error",
                    str(e),
                )

    def update_dash(self):
        for i in self.table.get_children():
            self.table.delete(i)

        for _, r in self.results_df.iterrows():
            model_name = r["Model"]

            train_f1 = r.get(
                "Train Macro F1",
                np.nan,
            )
            val_f1 = r.get(
                "Validation Macro F1",
                np.nan,
            )
            test_f1 = r.get(
                "Test Macro F1",
                np.nan,
            )
            val_acc = r.get(
                "Validation Accuracy",
                np.nan,
            )
            test_acc = r.get(
                "Test Accuracy",
                np.nan,
            )
            gap = r.get(
                "Overfit Gap",
                np.nan,
            )

            status = str(
                r.get(
                    "Overfit Status",
                    "—",
                )
            )

            vals = (
                model_name,
                "—" if pd.isna(train_f1) else f"{train_f1:.4f}",
                "—" if pd.isna(val_f1) else f"{val_f1:.4f}",
                "—" if pd.isna(test_f1) else f"{test_f1:.4f}",
                "—" if pd.isna(val_acc) else f"{val_acc:.4f}",
                "—" if pd.isna(test_acc) else f"{test_acc:.4f}",
                "—" if pd.isna(gap) else f"{gap:.4f}",
                status,
            )

            if model_name == self.best_model_name:
                tag = "winner"
            elif status == "HIGH":
                tag = "overfit"
            elif status == "WARNING":
                tag = "warning"
            else:
                tag = ()

            self.table.insert(
                "",
                "end",
                values=vals,
                tags=(tag,) if tag else (),
            )

        m = self.best_model_metrics

        self.best.set(
            self.best_model_name
        )

        self.acc.set(
            f"{m['Test Accuracy']:.4f}"
        )

        self.f1.set(
            f"{m['Test Macro F1']:.4f}"
        )

        self.bacc.set(
            f"{m['Test Balanced Accuracy']:.4f}"
        )

        self.gap.set(
            f"{m['Overfit Gap']:.4f}"
        )

        self.samples.set(
            f"{len(self.df):,}"
        )

    def update_dash_loaded(self):
        m = self.best_model_metrics

        self.best.set(
            self.best_model_name
        )

        # Older models may not have test metrics.
        acc = m.get(
            "Test Accuracy",
            m.get("Accuracy", 0),
        )
        f1 = m.get(
            "Test Macro F1",
            m.get("Macro F1", 0),
        )
        bacc = m.get(
            "Test Balanced Accuracy",
            m.get("Balanced Accuracy", 0),
        )
        gap = m.get(
            "Overfit Gap",
            np.nan,
        )

        self.acc.set(
            f"{acc:.4f}"
        )
        self.f1.set(
            f"{f1:.4f}"
        )
        self.bacc.set(
            f"{bacc:.4f}"
        )
        self.gap.set(
            "—"
            if pd.isna(gap)
            else f"{gap:.4f}"
        )

    def clear(self):
        for i in self.table.get_children():
            self.table.delete(i)

        for v in (
            self.best,
            self.acc,
            self.f1,
            self.bacc,
            self.samples,
            self.gap,
        ):
            v.set("—")

        self.status.set("Ready")
        self.log(
            "Dashboard display cleared. Trained model registry and "
            "production model were kept."
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self):
        if (
            self.best_model is None
            and os.path.exists(MODEL_PATH)
        ):
            try:
                b = joblib.load(MODEL_PATH)

                self.best_model = b["model"]
                self.best_model_name = b["best_model_name"]
                self.best_model_metrics = b[
                    "best_model_metrics"
                ]
                self.preprocessor = b[
                    "preprocessor"
                ]
                self.label_encoder = b[
                    "label_encoder"
                ]
                self.features = b[
                    "features"
                ]

            except Exception as e:
                messagebox.showerror(
                    "Model Error",
                    str(e),
                )
                return

        if self.best_model is None:
            messagebox.showwarning(
                "No Model",
                "Train or load a model first.",
            )
            return

        try:
            numeric = {
                "Age",
                "Height_cm",
                "Weight_kg",
                "BMI",
                "Stress_Level_0_10",
                "Mood_Score_0_10",
                "Heart_Rate_bpm",
                "SpO2_percent",
                "Sleep_Hours",
                "Daily_Steps",
                "Exercise_Minutes_Week",
                "AQI",
                "Humidity_percent",
                "UV_Index",
            }

            person = {}

            for k, v in self.pvars.items():
                if k in numeric:
                    person[k] = float(v.get())
                else:
                    person[k] = v.get()

            inp = pd.DataFrame([person])

            for f in self.features:
                if f not in inp.columns:
                    inp[f] = np.nan

            inp = inp[self.features]

            z = self.preprocessor.transform(inp)

            # HistGradientBoosting requires dense input.
            if self.best_model_name == "HistGradientBoosting":
                z = self._dense_if_needed(z)

            pred = self.best_model.predict(z)[0]

            risk = self.label_encoder.inverse_transform(
                [pred]
            )[0]

            if hasattr(
                self.best_model,
                "predict_proba",
            ):
                probs = self.best_model.predict_proba(z)[0]
                names = self.label_encoder.classes_

                lines = [
                    f"{n}: {p * 100:.2f}%"
                    for n, p in zip(
                        names,
                        probs,
                    )
                ]
            else:
                lines = [
                    "Probability output is not available "
                    "for this model."
                ]

            self.risk.config(
                text=risk,
                fg=(
                    RED
                    if risk == "High"
                    else YELLOW
                    if risk == "Moderate"
                    else ACCENT
                ),
            )

            gap = self.best_model_metrics.get(
                "Overfit Gap",
                np.nan,
            )

            gap_text = (
                "N/A"
                if pd.isna(gap)
                else f"{gap:.4f}"
            )

            self.pdetail.config(
                text=(
                    f"Model: {self.best_model_name}\n\n"
                    + "\n".join(lines)
                    + f"\n\nOverfit gap: {gap_text}"
                    + "\n\nAI wellness risk assessment only "
                      "— not a medical diagnosis."
                )
            )

            self.log(
                "\nHEALTH RISK PREDICTION"
            )
            self.log(
                f"Model: {self.best_model_name} "
                f"| Risk: {risk}"
            )

            for x in lines:
                self.log(x)

        except Exception as e:
            messagebox.showerror(
                "Prediction Error",
                str(e),
            )


if __name__ == "__main__":
    AegisAIApp().mainloop()
