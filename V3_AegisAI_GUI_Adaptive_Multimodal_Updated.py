
# ============================================================
# AegisAI — Modern Health & Wellness Risk Assessment
# Desktop GUI for the trained AegisAI ML model
#
# Requirements:
#   pip install customtkinter pandas numpy joblib scikit-learn xgboost matplotlib
#
# Put these files in the same folder:
#   AegisAI_GUI.py
#   AegisAI_Health_Risk_Model.joblib
# ============================================================

import os
import sys
import subprocess
import time
import math
import json
import urllib.parse
import urllib.request
import threading
import joblib
import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -----------------------------
# App configuration
# -----------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_NAME = "AegisAI"
APP_SUBTITLE = "Adaptive Multimodal AI for Personalized Healthcare & Wellness Risk Assessment"
FORMAL_PROJECT_TITLE = "Adaptive Multimodal AI for Personalized Healthcare & Wellness Risk Assessment"
PROJECT_DOMAIN = "Artificial Intelligence, Healthcare & Wellness, Machine Learning"
STUDENT_NAME = "Mayank Singh"
DEGREE = "B.Tech in Computer Science & Engineering (AI–ML)"
UNIVERSITY = "Galgotias University"
ACADEMIC_PERIOD = "August 2023 – June 2027"
ACADEMIC_YEAR = "2026–2027"
CGPA = "8.17 / 10"
DATASET_SIZE = "400,000 balanced synthetic research records"
MODEL_FILENAME = "AegisAI_Health_Risk_Model.joblib"
REALTIME_EMOTION_APP = "realtime_emotion_app.py"

BG = "#080D17"
SIDEBAR = "#0D1422"
CARD = "#111A2A"
CARD_2 = "#162236"
BORDER = "#24334B"
TEXT = "#F4F7FB"
MUTED = "#8E9BB0"
ACCENT = "#38BDF8"
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
DANGER = "#EF4444"
PURPLE = "#8B5CF6"


class AegisAIApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("AegisAI — Adaptive Multimodal AI for Personalized Healthcare & Wellness Risk Assessment")
        self.geometry("1440x900")
        self.minsize(1180, 760)
        self.configure(fg_color=BG)

        self.model_bundle = None
        self.model_loaded = False
        self.current_result = None

        # Live environment/weather state
        self.environment_fetching = False
        self.environment_location = None
        self.environment_last_update = None

        self.vars = {}
        self.entries = {}

        self._load_model()
        self._build_layout()
        self._build_dashboard()
        self._build_assessment()
        self._build_results()

        # Automatically populate live environmental conditions once the GUI is ready.
        self.after(700, self.fetch_live_environment)
        self._build_about()

        self.show_page("dashboard")

    # ========================================================
    # MODEL
    # ========================================================

    def _model_path(self):
        # Always resolve the model relative to this Python file, not the
        # PowerShell/current working directory. This also handles spaces in
        # Windows folder names safely.
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            MODEL_FILENAME
        )

    def _update_model_status(self):
        """Update the visible model-status indicator after load/reload/import."""
        label = getattr(self, "model_status_label", None)
        if label is None:
            return

        if self.model_loaded:
            label.configure(text="●  Model Ready", text_color=SUCCESS)
        else:
            label.configure(text="●  Model Missing", text_color=DANGER)

    def _load_model(self):
        try:
            model_path = self._model_path()

            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"{MODEL_FILENAME} was not found in:\n{os.path.dirname(model_path)}"
                )

            self.model_bundle = joblib.load(model_path)
            self.model_loaded = True
            self._update_model_status()

        except Exception as e:
            self.model_loaded = False
            self.model_bundle = None
            self._update_model_status()
            print("Model loading error:", e)

    def import_custom_model(self):
        """Import a custom .joblib model from the user's PC."""
        selected = filedialog.askopenfilename(
            title="Import Custom AegisAI Model",
            filetypes=[
                ("Joblib Model", "*.joblib"),
                ("All Files", "*.*")
            ]
        )

        if not selected:
            return

        try:
            import shutil
            model_path = self._model_path()

            # Copy the selected custom model into the application's model path.
            shutil.copy2(selected, model_path)

            # Immediately load/check the imported model.
            self._load_model()

            if self.model_loaded:
                messagebox.showinfo(
                    "Custom Model Imported",
                    "Custom model imported successfully.\n\n"
                    f"Model path:\n{model_path}"
                )
            else:
                raise RuntimeError("The selected model could not be loaded by AegisAI.")

        except Exception as e:
            messagebox.showerror(
                "Custom Model Import Error",
                f"Could not import the selected model.\n\n{e}"
            )

    def refresh_application(self):
        """Re-check the model path and reload the model without restarting the GUI."""
        try:
            model_path = self._model_path()

            # Refresh only re-checks the expected model path and reloads it.
            # It does NOT restart/close the application.
            self._load_model()

            if self.model_loaded:
                messagebox.showinfo(
                    "Model Refreshed",
                    "Model path checked successfully and the model was reloaded.\n\n"
                    f"Path:\n{model_path}"
                )
            else:
                messagebox.showwarning(
                    "Model Not Found",
                    "AegisAI could not load the model from the expected path.\n\n"
                    f"Expected path:\n{model_path}\n\n"
                    "Use 'Import Custom Model' to select your model from the PC."
                )

        except Exception as e:
            messagebox.showerror(
                "Refresh Error",
                f"Could not refresh the model.\n\n{e}"
            )

    # ========================================================
    # GENERAL UI
    # ========================================================

    def _build_layout(self):

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(
            self,
            width=250,
            corner_radius=0,
            fg_color=SIDEBAR
        )
        self.sidebar.grid(
            row=0,
            column=0,
            sticky="nsew"
        )
        self.sidebar.grid_propagate(False)

        # Main content
        self.content = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color=BG
        )
        self.content.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self._build_sidebar()

        # Always-visible application exit control.
        self.exit_button = ctk.CTkButton(
            self.content,
            text="✕  Exit",
            width=88,
            height=34,
            corner_radius=9,
            fg_color=DANGER,
            hover_color="#DC2626",
            text_color=TEXT,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._safe_exit
        )
        self.exit_button.place(relx=1.0, x=-18, y=14, anchor="ne")

    def _safe_exit(self):
        """Close the application after a final user confirmation."""
        try:
            if messagebox.askyesno("Exit AegisAI", "Are you sure you want to exit AegisAI?"):
                self.destroy()
        except Exception:
            self.destroy()

    def _update_clock(self):

        if hasattr(self, "clock_label") and self.clock_label.winfo_exists():
            self.clock_label.configure(
                text=time.strftime("%d %b %Y  •  %I:%M:%S %p")
            )
            self.after(1000, self._update_clock)

    def open_realtime_emotion_app(self):

        """Launch the facial-expression/voice application from the same folder."""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            app_path = os.path.join(base_dir, REALTIME_EMOTION_APP)

            if not os.path.isfile(app_path):
                messagebox.showerror(
                    "File Not Found",
                    f"{REALTIME_EMOTION_APP} was not found in the same folder as AegisAI."
                )
                return

            subprocess.Popen(
                [sys.executable, app_path],
                cwd=base_dir
            )

        except Exception as e:
            messagebox.showerror(
                "Launch Error",
                f"Could not start {REALTIME_EMOTION_APP}.\n\n{e}"
            )

    def _build_sidebar(self):

        logo = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )
        logo.pack(
            fill="x",
            padx=24,
            pady=(28, 24)
        )

        logo_icon = ctk.CTkLabel(
            logo,
            text="✦",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=ACCENT
        )
        logo_icon.pack(side="left")

        logo_text = ctk.CTkLabel(
            logo,
            text=" AegisAI",
            font=ctk.CTkFont(size=23, weight="bold"),
            text_color=TEXT
        )
        logo_text.pack(side="left")

        self.nav_buttons = {}

        self._nav_button(
            "dashboard",
            "⌂",
            "Dashboard"
        )

        self._nav_button(
            "assessment",
            "✚",
            "New Assessment"
        )

        self._nav_button(
            "results",
            "◉",
            "Assessment Result"
        )

        self._nav_button(
            "about",
            "ⓘ",
            "About AegisAI"
        )

        # Spacer
        ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        ).pack(fill="both", expand=True)

        # --------------------------------------------------------
        # Application controls
        # --------------------------------------------------------
        controls = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )
        controls.pack(fill="x", padx=14, pady=(0, 12))

        ctk.CTkButton(
            controls,
            text="↻  Refresh Model",
            height=40,
            corner_radius=10,
            fg_color=CARD_2,
            hover_color="#24334B",
            text_color=TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.refresh_application
        ).pack(fill="x")

        ctk.CTkButton(
            controls,
            text="＋  Import Custom Model",
            height=40,
            corner_radius=10,
            fg_color=ACCENT,
            hover_color="#0EA5E9",
            text_color="#06111F",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.import_custom_model
        ).pack(fill="x", pady=(8, 0))

        self.clock_label = ctk.CTkLabel(
            controls,
            text="",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=ACCENT
        )
        self.clock_label.pack(pady=(9, 0))
        self._update_clock()

        # Model status card
        status_card = ctk.CTkFrame(
            self.sidebar,
            fg_color=CARD,
            corner_radius=16,
            border_width=1,
            border_color=BORDER
        )
        status_card.pack(
            fill="x",
            padx=18,
            pady=18
        )

        ctk.CTkLabel(
            status_card,
            text="MODEL STATUS",
            font=ctk.CTkFont(
                size=10,
                weight="bold"
            ),
            text_color=MUTED
        ).pack(
            anchor="w",
            padx=15,
            pady=(14, 4)
        )

        status = "●  Model Ready" if self.model_loaded else "●  Model Missing"
        color = SUCCESS if self.model_loaded else DANGER

        self.model_status_label = ctk.CTkLabel(
            status_card,
            text=status,
            font=ctk.CTkFont(
                size=14,
                weight="bold"
            ),
            text_color=color
        )
        self.model_status_label.pack(
            anchor="w",
            padx=15,
            pady=(0, 14)
        )

        ctk.CTkLabel(
            self.sidebar,
            text="AegisAI v2.0  •  Adaptive Multimodal AI",
            font=ctk.CTkFont(size=10),
            text_color=MUTED
        ).pack(
            pady=(0, 18)
        )

    def _nav_button(self, page, icon, text):

        btn = ctk.CTkButton(
            self.sidebar,
            text=f"  {icon}   {text}",
            anchor="w",
            height=48,
            corner_radius=12,
            fg_color="transparent",
            hover_color=CARD_2,
            text_color=MUTED,
            font=ctk.CTkFont(
                size=14,
                weight="bold"
            ),
            command=lambda p=page: self.show_page(p)
        )

        btn.pack(
            fill="x",
            padx=14,
            pady=4
        )

        self.nav_buttons[page] = btn

    def _page_frame(self):

        frame = ctk.CTkFrame(
            self.content,
            fg_color=BG,
            corner_radius=0
        )

        frame.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        return frame

    def show_page(self, page):

        for widget in self.content.winfo_children():
            widget.grid_forget()

        for key, btn in self.nav_buttons.items():
            if key == page:
                btn.configure(
                    fg_color=CARD_2,
                    text_color=TEXT
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=MUTED
                )

        target = getattr(
            self,
            f"{page}_page",
            None
        )

        if target:
            target.grid(
                row=0,
                column=0,
                sticky="nsew"
            )

    # ========================================================
    # COMMON COMPONENTS
    # ========================================================

    def _header(self, parent, title, subtitle):

        header = ctk.CTkFrame(
            parent,
            fg_color="transparent"
        )
        header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=32,
            pady=(28, 14)
        )

        ctk.CTkLabel(
            header,
            text=title,
            font=ctk.CTkFont(
                size=30,
                weight="bold"
            ),
            text_color=TEXT
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text=subtitle,
            font=ctk.CTkFont(
                size=13
            ),
            text_color=MUTED
        ).pack(
            anchor="w",
            pady=(4, 0)
        )

        return header

    def _card(self, parent, **kwargs):

        return ctk.CTkFrame(
            parent,
            fg_color=CARD,
            corner_radius=18,
            border_width=1,
            border_color=BORDER,
            **kwargs
        )

    def _section_title(self, parent, title, subtitle=None):

        frame = ctk.CTkFrame(
            parent,
            fg_color="transparent"
        )
        frame.pack(
            fill="x",
            padx=18,
            pady=(16, 10)
        )

        ctk.CTkLabel(
            frame,
            text=title,
            font=ctk.CTkFont(
                size=16,
                weight="bold"
            ),
            text_color=TEXT
        ).pack(anchor="w")

        if subtitle:
            ctk.CTkLabel(
                frame,
                text=subtitle,
                font=ctk.CTkFont(
                    size=11
                ),
                text_color=MUTED
            ).pack(
                anchor="w",
                pady=(2, 0)
            )

        return frame

    def _field(
        self,
        parent,
        key,
        label,
        default="",
        placeholder="",
        width=220
    ):

        frame = ctk.CTkFrame(
            parent,
            fg_color="transparent"
        )

        frame.pack(
            side="left",
            fill="x",
            expand=True,
            padx=8,
            pady=7
        )

        ctk.CTkLabel(
            frame,
            text=label,
            font=ctk.CTkFont(
                size=11,
                weight="bold"
            ),
            text_color=MUTED
        ).pack(
            anchor="w",
            pady=(0, 5)
        )

        var = ctk.StringVar(
            value=str(default)
        )

        entry = ctk.CTkEntry(
            frame,
            textvariable=var,
            placeholder_text=placeholder,
            height=38,
            corner_radius=9,
            fg_color=CARD_2,
            border_color=BORDER,
            text_color=TEXT,
            placeholder_text_color="#5E6A7E",
            width=width
        )

        entry.pack(
            fill="x"
        )

        self.vars[key] = var
        self.entries[key] = entry

        return entry

    def _select(
        self,
        parent,
        key,
        label,
        values,
        default=None
    ):

        frame = ctk.CTkFrame(
            parent,
            fg_color="transparent"
        )

        frame.pack(
            side="left",
            fill="x",
            expand=True,
            padx=8,
            pady=7
        )

        ctk.CTkLabel(
            frame,
            text=label,
            font=ctk.CTkFont(
                size=11,
                weight="bold"
            ),
            text_color=MUTED
        ).pack(
            anchor="w",
            pady=(0, 5)
        )

        value = default or values[0]

        var = ctk.StringVar(
            value=value
        )

        menu = ctk.CTkOptionMenu(
            frame,
            variable=var,
            values=values,
            height=38,
            corner_radius=9,
            fg_color=CARD_2,
            button_color=CARD_2,
            button_hover_color="#1D2C43",
            dropdown_fg_color=CARD_2,
            text_color=TEXT,
            width=220
        )

        menu.pack(
            fill="x"
        )

        self.vars[key] = var

        return menu

    # ========================================================
    # DASHBOARD
    # ========================================================

    def _build_dashboard(self):

        self.dashboard_page = self._page_frame()

        self._header(
            self.dashboard_page,
            "Adaptive Multimodal AI",
            "Personalized Healthcare & Wellness Risk Assessment"
        )

        body = ctk.CTkFrame(
            self.dashboard_page,
            fg_color="transparent"
        )

        body.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=32,
            pady=(0, 30)
        )

        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)

        # --------------------------------------------------------
        # Project hero
        # --------------------------------------------------------
        hero = self._card(body)
        hero.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 18)
        )

        hero.grid_columnconfigure(0, weight=1)

        left = ctk.CTkFrame(
            hero,
            fg_color="transparent"
        )
        left.grid(
            row=0,
            column=0,
            sticky="w",
            padx=28,
            pady=26
        )

        ctk.CTkLabel(
            left,
            text="AEGISAI  •  FINAL-YEAR AI PROJECT",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT
        ).pack(anchor="w")

        ctk.CTkLabel(
            left,
            text=FORMAL_PROJECT_TITLE,
            font=ctk.CTkFont(size=25, weight="bold"),
            text_color=TEXT,
            wraplength=820,
            justify="left"
        ).pack(anchor="w", pady=(7, 0))

        ctk.CTkLabel(
            left,
            text="From Multiple Signals to Smarter Health Decisions.",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=PURPLE
        ).pack(anchor="w", pady=(8, 0))

        ctk.CTkLabel(
            left,
            text=(
                "An adaptive multimodal AI research prototype integrating "
                "facial expression, voice, symptoms, physiological signals, "
                "lifestyle, nutrition, medical context and environmental factors "
                "for personalized wellness-risk assessment."
            ),
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            wraplength=850,
            justify="left"
        ).pack(anchor="w", pady=(10, 16))

        ctk.CTkButton(
            left,
            text="✚   Start New Assessment",
            height=44,
            corner_radius=11,
            fg_color=ACCENT,
            hover_color="#0EA5E9",
            text_color="#03131D",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda: self.show_page("assessment")
        ).pack(anchor="w")

        # --------------------------------------------------------
        # Student / project snapshot
        # --------------------------------------------------------
        snapshot = self._card(body)
        snapshot.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(0, 10)
        )

        self._section_title(
            snapshot,
            "Project & Student Profile",
            "Academic and technical context"
        )

        profile = [
            ("STUDENT", STUDENT_NAME),
            ("PROGRAM", DEGREE),
            ("UNIVERSITY", UNIVERSITY),
            ("PERIOD", ACADEMIC_PERIOD),
            ("CGPA", CGPA),
            ("DOMAIN", PROJECT_DOMAIN),
        ]

        for title, value in profile:
            row = ctk.CTkFrame(
                snapshot,
                fg_color=CARD_2,
                corner_radius=10
            )
            row.pack(fill="x", padx=18, pady=4)

            ctk.CTkLabel(
                row,
                text=title,
                text_color=MUTED,
                font=ctk.CTkFont(size=9, weight="bold")
            ).pack(anchor="w", padx=12, pady=(7, 0))

            ctk.CTkLabel(
                row,
                text=value,
                text_color=TEXT,
                font=ctk.CTkFont(size=12, weight="bold"),
                wraplength=650,
                justify="left"
            ).pack(anchor="w", padx=12, pady=(0, 7))

        # --------------------------------------------------------
        # Model snapshot
        # --------------------------------------------------------
        metrics = self._card(body)
        metrics.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=(10, 0)
        )

        self._section_title(
            metrics,
            "Model Snapshot",
            "Current application configuration"
        )

        metric_data = [
            ("MODEL", "XGBoost"),
            ("TASK", "3-Class Wellness Risk"),
            ("DATA", DATASET_SIZE),
            ("OUTPUT", "Low / Moderate / High"),
            ("ENGINE", "ML Model + Safety Layer"),
            ("MODE", "Adaptive Multimodal"),
        ]

        for title, value in metric_data:
            row = ctk.CTkFrame(
                metrics,
                fg_color=CARD_2,
                corner_radius=10
            )
            row.pack(fill="x", padx=18, pady=4)

            ctk.CTkLabel(
                row,
                text=title,
                text_color=MUTED,
                font=ctk.CTkFont(size=9, weight="bold")
            ).pack(anchor="w", padx=12, pady=(7, 0))

            ctk.CTkLabel(
                row,
                text=value,
                text_color=TEXT,
                font=ctk.CTkFont(size=12, weight="bold"),
                wraplength=330,
                justify="left"
            ).pack(anchor="w", padx=12, pady=(0, 7))

    # ========================================================
    # ASSESSMENT
    # ========================================================

    def _build_assessment(self):

        self.assessment_page = self._page_frame()

        self._header(
            self.assessment_page,
            "New Health Assessment",
            "Enter the available information for the person being assessed"
        )

        scroll = ctk.CTkScrollableFrame(
            self.assessment_page,
            fg_color="transparent",
            scrollbar_button_color=CARD_2,
            scrollbar_button_hover_color="#263B58"
        )

        scroll.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=25,
            pady=(0, 10)
        )

        # -------------------------
        # Personal
        # -------------------------
        card = self._card(scroll)
        card.pack(
            fill="x",
            padx=8,
            pady=8
        )

        self._section_title(
            card,
            "01  Personal & Medical Context",
            "Basic demographic and medical background"
        )

        row = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row.pack(fill="x", padx=10)

        self._field(row, "Age", "Age", 22)
        self._select(
            row,
            "Sex",
            "Sex",
            ["Male", "Female"],
            "Male"
        )
        self._field(row, "Height_cm", "Height (cm)", 175)
        self._field(row, "Weight_kg", "Weight (kg)", 68)

        row2 = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row2.pack(fill="x", padx=10)

        self._field(
            row2,
            "BMI",
            "BMI",
            22.2
        )
        self._select(
            row2,
            "Existing_Condition",
            "Existing Condition",
            [
                "None",
                "Diabetes",
                "Hypertension",
                "Asthma",
                "Heart Disease",
                "Thyroid"
            ],
            "None"
        )
        self._select(
            row2,
            "Medication_Status",
            "Medication",
            ["None", "Regular", "Irregular"],
            "None"
        )
        self._select(
            row2,
            "Family_History",
            "Family History",
            [
                "None",
                "Diabetes",
                "Hypertension",
                "Heart Disease",
                "Multiple"
            ],
            "None"
        )

        # -------------------------
        # Face + voice
        # -------------------------
        card = self._card(scroll)
        card.pack(fill="x", padx=8, pady=8)

        self._section_title(
            card,
            "02  Facial Expression & Voice",
            "Multimodal emotional-state indicators"
        )

        ctk.CTkButton(
            card,
            text="◉  Open Real-Time Facial & Voice Analysis",
            height=40,
            corner_radius=10,
            fg_color=PURPLE,
            hover_color="#7C3AED",
            text_color=TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.open_realtime_emotion_app
        ).pack(fill="x", padx=18, pady=(0, 10))

        row = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row.pack(fill="x", padx=10)

        self._select(
            row,
            "Facial_Expression",
            "Facial Expression",
            [
                "Happy",
                "Sad",
                "Angry",
                "Fear",
                "Neutral"
            ],
            "Neutral"
        )

        self._field(
            row,
            "Stress_Level_0_10",
            "Stress (0–10)",
            3
        )

        self._field(
            row,
            "Mood_Score_0_10",
            "Mood Score (0–10)",
            8
        )

        self._field(
            row,
            "Voice_Stress_0_10",
            "Voice Stress (0–10)",
            3
        )

        row2 = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row2.pack(fill="x", padx=10)

        self._field(
            row2,
            "Voice_Tone_Score_0_10",
            "Voice Tone (0–10)",
            8
        )

        # -------------------------
        # Vitals
        # -------------------------
        card = self._card(scroll)
        card.pack(fill="x", padx=8, pady=8)

        self._section_title(
            card,
            "03  Vital Signs & Blood Pressure",
            "Core physiological measurements"
        )

        row = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row.pack(fill="x", padx=10)

        self._field(
            row,
            "Systolic_BP_mmHg",
            "Systolic BP (mmHg)",
            118
        )

        self._field(
            row,
            "Diastolic_BP_mmHg",
            "Diastolic BP (mmHg)",
            76
        )

        self._field(
            row,
            "Heart_Rate_bpm",
            "Heart Rate (bpm)",
            68
        )

        self._field(
            row,
            "SpO2_percent",
            "SpO₂ (%)",
            98
        )

        row2 = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row2.pack(fill="x", padx=10)

        self._field(
            row2,
            "Body_Temperature_C",
            "Body Temperature (°C)",
            36.7
        )

        self._field(
            row2,
            "Respiratory_Rate_bpm",
            "Respiratory Rate",
            15
        )

        # -------------------------
        # Symptoms
        # -------------------------
        card = self._card(scroll)
        card.pack(fill="x", padx=8, pady=8)

        self._section_title(
            card,
            "04  Symptoms & Current State",
            "Current symptoms reported by the person"
        )

        row = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row.pack(fill="x", padx=10)

        self._field(
            row,
            "Fatigue_Score_0_10",
            "Fatigue (0–10)",
            2
        )

        self._field(
            row,
            "Symptom_Count",
            "Symptom Count",
            0
        )

        self._select(
            row,
            "Primary_Symptom",
            "Primary Symptom",
            [
                "None",
                "Headache",
                "Fatigue",
                "Cough",
                "Dizziness",
                "Body Pain",
                "Breathlessness",
                "Nausea"
            ],
            "None"
        )

        # -------------------------
        # Lifestyle
        # -------------------------
        card = self._card(scroll)
        card.pack(fill="x", padx=8, pady=8)

        self._section_title(
            card,
            "05  Lifestyle & Activity",
            "Sleep, exercise and daily habits"
        )

        row = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row.pack(fill="x", padx=10)

        self._field(
            row,
            "Sleep_Hours",
            "Sleep (hours)",
            7.5
        )

        self._field(
            row,
            "Daily_Steps",
            "Daily Steps",
            8500
        )

        self._field(
            row,
            "Exercise_Minutes_Week",
            "Exercise / week (min)",
            180
        )

        self._field(
            row,
            "Sedentary_Hours_Day",
            "Sedentary (hours/day)",
            6
        )

        row2 = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row2.pack(fill="x", padx=10)

        self._select(
            row2,
            "Smoking_Status",
            "Smoking",
            ["Never", "Former", "Current"],
            "Never"
        )

        self._select(
            row2,
            "Alcohol_Frequency",
            "Alcohol",
            ["None", "Occasional", "Frequent"],
            "None"
        )

        # -------------------------
        # Diet
        # -------------------------
        card = self._card(scroll)
        card.pack(fill="x", padx=8, pady=8)

        self._section_title(
            card,
            "06  Diet & Hydration",
            "Nutritional and hydration indicators"
        )

        row = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row.pack(fill="x", padx=10)

        self._field(
            row,
            "Water_Liters_Day",
            "Water (L/day)",
            2.5
        )

        self._field(
            row,
            "Calories_Day",
            "Calories/day",
            2200
        )

        self._field(
            row,
            "Protein_g_Day",
            "Protein (g/day)",
            80
        )

        self._field(
            row,
            "Sugar_g_Day",
            "Sugar (g/day)",
            40
        )

        row2 = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row2.pack(fill="x", padx=10)

        self._field(
            row2,
            "Fruit_Veg_Servings_Day",
            "Fruit & Veg servings/day",
            5
        )

        # -------------------------
        # Environment
        # -------------------------
        card = self._card(scroll)
        card.pack(fill="x", padx=8, pady=8)

        self._section_title(
            card,
            "07  Surrounding Environment",
            "Live location-based weather & air-quality data"
        )

        # --------------------------------------------------------
        # Live environment controls
        # --------------------------------------------------------
        location_bar = ctk.CTkFrame(
            card,
            fg_color=CARD_2,
            corner_radius=12
        )
        location_bar.pack(fill="x", padx=18, pady=(0, 10))

        ctk.CTkLabel(
            location_bar,
            text="LOCATION",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=MUTED
        ).pack(side="left", padx=(12, 6), pady=10)

        self.environment_location_label = ctk.CTkLabel(
            location_bar,
            text="Detecting location…",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT,
            anchor="w"
        )
        self.environment_location_label.pack(side="left", fill="x", expand=True, padx=4)

        self.environment_fetch_button = ctk.CTkButton(
            location_bar,
            text="↻  Fetch Live Data",
            height=32,
            width=150,
            corner_radius=8,
            fg_color=ACCENT,
            hover_color="#0EA5E9",
            text_color="#03131D",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self.fetch_live_environment
        )
        self.environment_fetch_button.pack(side="right", padx=10, pady=6)

        self.environment_status_label = ctk.CTkLabel(
            card,
            text="Live environment data will be loaded automatically from your location.",
            font=ctk.CTkFont(size=10),
            text_color=MUTED,
            anchor="w"
        )
        self.environment_status_label.pack(fill="x", padx=20, pady=(0, 5))

        row = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row.pack(fill="x", padx=10)

        self._field(
            row,
            "AQI",
            "Air Quality Index",
            80
        )

        self._field(
            row,
            "PM2_5_ug_m3",
            "PM2.5 (µg/m³)",
            30
        )

        self._field(
            row,
            "Temperature_Environment_C",
            "Ambient Temp (°C)",
            27
        )

        self._field(
            row,
            "Humidity_percent",
            "Humidity (%)",
            55
        )

        row2 = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        row2.pack(fill="x", padx=10)

        self._field(
            row2,
            "Noise_dB",
            "Noise (dB) — manual",
            50
        )

        self._field(
            row2,
            "UV_Index",
            "UV Index",
            5
        )

        # -------------------------
        # Action bar
        # -------------------------
        action = ctk.CTkFrame(
            scroll,
            fg_color="transparent"
        )
        action.pack(
            fill="x",
            padx=8,
            pady=20
        )

        ctk.CTkButton(
            action,
            text="↻  Reset",
            height=48,
            width=150,
            corner_radius=12,
            fg_color=CARD_2,
            hover_color="#24334B",
            text_color=TEXT,
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            ),
            command=self.reset_form
        ).pack(
            side="left"
        )

        ctk.CTkButton(
            action,
            text="⚡  Analyze Wellness Risk",
            height=48,
            width=260,
            corner_radius=12,
            fg_color=ACCENT,
            hover_color="#0EA5E9",
            text_color="#03131D",
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            ),
            command=self.run_prediction
        ).pack(
            side="right"
        )

    # ========================================================
    # LIVE WEATHER / AIR QUALITY
    # ========================================================

    @staticmethod
    def _http_json(url, timeout=12):
        """Fetch JSON without adding an external requests dependency."""
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AegisAI/2.0"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _detect_public_ip_location(self):
        """Resolve an approximate location from the public IP address.

        Public-IP geolocation is approximate; it is not GPS-level positioning.
        """
        data = self._http_json("https://ipapi.co/json/")
        if not data.get("latitude") or not data.get("longitude"):
            raise RuntimeError("Could not determine latitude/longitude from the public IP.")

        return {
            "latitude": float(data["latitude"]),
            "longitude": float(data["longitude"]),
            "city": data.get("city") or "Unknown city",
            "region": data.get("region") or "",
            "country": data.get("country_name") or "",
        }

    def _fetch_open_meteo_environment(self, latitude, longitude):
        """Fetch current weather + air quality for the resolved coordinates."""
        weather_params = urllib.parse.urlencode({
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,rain,weather_code,cloud_cover,wind_speed_10m,"
                "surface_pressure,uv_index"
            ),
            "timezone": "auto",
            "forecast_days": 1,
        })
        air_params = urllib.parse.urlencode({
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,"
                "sulphur_dioxide,ozone,us_aqi,european_aqi,uv_index"
            ),
            "timezone": "auto",
            "forecast_days": 1,
        })

        weather = self._http_json(
            f"https://api.open-meteo.com/v1/forecast?{weather_params}"
        )
        air = self._http_json(
            f"https://air-quality-api.open-meteo.com/v1/air-quality?{air_params}"
        )

        wc = weather.get("current", {})
        ac = air.get("current", {})

        def required_number(source, key):
            value = source.get(key)
            if value is None:
                raise RuntimeError(f"Open-Meteo did not return '{key}'.")
            return float(value)

        # Prefer the combined US AQI for the model's AQI field because the
        # model field is a 0-1000 numeric AQI input and US AQI is explicitly
        # supplied by the air-quality API.
        us_aqi = ac.get("us_aqi")
        eu_aqi = ac.get("european_aqi")
        aqi = float(us_aqi if us_aqi is not None else eu_aqi)

        return {
            "AQI": round(aqi),
            "PM2_5_ug_m3": round(required_number(ac, "pm2_5"), 2),
            "Temperature_Environment_C": round(required_number(wc, "temperature_2m"), 2),
            "Humidity_percent": round(required_number(wc, "relative_humidity_2m"), 2),
            "UV_Index": round(float(wc.get("uv_index", ac.get("uv_index", 0))), 2),
            "apparent_temperature": round(float(wc.get("apparent_temperature", 0)), 2),
            "wind_speed": round(float(wc.get("wind_speed_10m", 0)), 2),
            "weather_code": wc.get("weather_code"),
            "cloud_cover": wc.get("cloud_cover"),
            "surface_pressure": wc.get("surface_pressure"),
            "precipitation": wc.get("precipitation"),
            "rain": wc.get("rain"),
            "pm10": ac.get("pm10"),
            "ozone": ac.get("ozone"),
            "no2": ac.get("nitrogen_dioxide"),
            "so2": ac.get("sulphur_dioxide"),
            "co": ac.get("carbon_monoxide"),
            "us_aqi": us_aqi,
            "european_aqi": eu_aqi,
        }

    def _environment_fetch_worker(self):
        try:
            location = self._detect_public_ip_location()
            environment = self._fetch_open_meteo_environment(
                location["latitude"], location["longitude"]
            )
            self.after(0, lambda: self._apply_live_environment(location, environment))
        except Exception as exc:
            self.after(0, lambda e=str(exc): self._environment_fetch_failed(e))

    def fetch_live_environment(self):
        """Start a non-blocking live environment refresh."""
        if self.environment_fetching:
            return

        self.environment_fetching = True
        if hasattr(self, "environment_fetch_button"):
            self.environment_fetch_button.configure(
                state="disabled",
                text="⟳  Fetching…"
            )
        if hasattr(self, "environment_status_label"):
            self.environment_status_label.configure(
                text="Fetching current location, weather and air-quality data…",
                text_color=ACCENT
            )

        threading.Thread(
            target=self._environment_fetch_worker,
            daemon=True
        ).start()

    def _apply_live_environment(self, location, env):
        self.environment_fetching = False
        self.environment_location = location
        self.environment_last_update = time.strftime("%d %b %Y %I:%M:%S %p")

        # These are the actual model input fields.
        live_values = {
            "AQI": str(env["AQI"]),
            "PM2_5_ug_m3": str(env["PM2_5_ug_m3"]),
            "Temperature_Environment_C": str(env["Temperature_Environment_C"]),
            "Humidity_percent": str(env["Humidity_percent"]),
            "UV_Index": str(env["UV_Index"]),
        }
        for key, value in live_values.items():
            if key in self.vars:
                self.vars[key].set(value)

        location_text = f"{location['city']}, {location['region']}"
        if location.get("country"):
            location_text += f", {location['country']}"
        location_text += f"  •  {location['latitude']:.4f}, {location['longitude']:.4f}"

        self.environment_location_label.configure(
            text=location_text,
            text_color=TEXT
        )
        self.environment_status_label.configure(
            text=(
                f"✓ Live data updated {self.environment_last_update}  •  "
                f"Temp {env['Temperature_Environment_C']:.1f}°C  •  "
                f"Humidity {env['Humidity_percent']:.0f}%  •  "
                f"AQI {env['AQI']}  •  PM2.5 {env['PM2_5_ug_m3']:.1f} µg/m³  •  "
                f"UV {env['UV_Index']:.1f}  •  "
                f"Wind {env['wind_speed']:.1f} km/h"
            ),
            text_color=SUCCESS
        )
        self.environment_fetch_button.configure(
            state="normal",
            text="↻  Refresh Live Data"
        )

    def _environment_fetch_failed(self, error):
        self.environment_fetching = False
        self.environment_fetch_button.configure(
            state="normal",
            text="↻  Fetch Live Data"
        )
        self.environment_status_label.configure(
            text=(
                "Live fetch failed. Check internet access and try again. "
                "Existing values were kept unchanged."
            ),
            text_color=DANGER
        )
        messagebox.showerror(
            "Live Environment Data Error",
            "Could not fetch live weather/air-quality data.\n\n"
            + error
            + "\n\n"
            "AQI, PM2.5, temperature, humidity and UV are not replaced with fake values."
        )

    # ========================================================
    # RESULTS
    # ========================================================

    def _build_results(self):

        self.results_page = self._page_frame()

        self._header(
            self.results_page,
            "Assessment Result",
            "Model-generated wellness risk estimate"
        )

        self.result_body = ctk.CTkFrame(
            self.results_page,
            fg_color="transparent"
        )

        self.result_body.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=32,
            pady=(0, 25)
        )

        self.result_body.grid_columnconfigure(
            0,
            weight=1
        )
        self.result_body.grid_columnconfigure(
            1,
            weight=1
        )
        self.result_body.grid_rowconfigure(
            0,
            weight=1
        )

        # Main result card
        self.result_card = self._card(
            self.result_body
        )
        self.result_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 10)
        )

        self.result_title = ctk.CTkLabel(
            self.result_card,
            text="No assessment yet",
            font=ctk.CTkFont(
                size=25,
                weight="bold"
            ),
            text_color=TEXT
        )
        self.result_title.pack(
            pady=(45, 10)
        )

        self.result_risk = ctk.CTkLabel(
            self.result_card,
            text="—",
            font=ctk.CTkFont(
                size=54,
                weight="bold"
            ),
            text_color=MUTED
        )
        self.result_risk.pack(
            pady=(0, 8)
        )

        self.result_confidence = ctk.CTkLabel(
            self.result_card,
            text="Run an assessment to see probabilities",
            font=ctk.CTkFont(
                size=13
            ),
            text_color=MUTED
        )
        self.result_confidence.pack()

        self.result_message = ctk.CTkLabel(
            self.result_card,
            text="",
            font=ctk.CTkFont(
                size=12
            ),
            text_color=MUTED,
            wraplength=480,
            justify="center"
        )
        self.result_message.pack(
            pady=(25, 12)
        )

        ctk.CTkButton(
            self.result_card,
            text="▣  Complete Report & Care Plan",
            height=42,
            corner_radius=10,
            fg_color=PURPLE,
            hover_color="#7C3AED",
            text_color=TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.show_complete_report
        ).pack(pady=(0, 8))

        ctk.CTkButton(
            self.result_card,
            text="←  New Assessment",
            height=42,
            corner_radius=10,
            fg_color=CARD_2,
            hover_color="#24334B",
            text_color=TEXT,
            command=lambda: self.show_page("assessment")
        ).pack(
            pady=10
        )

        # Probability card
        prob_card = self._card(
            self.result_body
        )
        prob_card.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(10, 0)
        )

        self._section_title(
            prob_card,
            "Risk Probability",
            "Model confidence by predicted class"
        )

        self.chart_frame = ctk.CTkFrame(
            prob_card,
            fg_color="transparent"
        )
        self.chart_frame.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10
        )

        self._draw_empty_chart()

    def _draw_empty_chart(self):

        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        fig = Figure(
            figsize=(5.2, 4.5),
            dpi=100,
            facecolor=CARD
        )

        ax = fig.add_subplot(111)
        ax.set_facecolor(CARD)

        ax.text(
            0.5,
            0.5,
            "Run an assessment\n\nto view probability",
            ha="center",
            va="center",
            color=MUTED,
            fontsize=13,
            transform=ax.transAxes
        )

        ax.axis("off")

        canvas = FigureCanvasTkAgg(
            fig,
            master=self.chart_frame
        )

        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    def _draw_chart(self, probabilities):

        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        labels = list(probabilities.keys())
        values = [
            probabilities[x] * 100
            for x in labels
        ]

        fig = Figure(
            figsize=(5.2, 4.5),
            dpi=100,
            facecolor=CARD
        )

        ax = fig.add_subplot(111)
        ax.set_facecolor(CARD)

        bars = ax.bar(
            labels,
            values,
            width=0.58,
            color=[
                DANGER,
                WARNING,
                SUCCESS
            ][:len(labels)]
        )

        ax.set_ylim(0, 100)
        ax.set_ylabel(
            "Probability (%)",
            color=MUTED
        )

        ax.tick_params(
            colors=MUTED,
            labelsize=10
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.grid(
            axis="y",
            alpha=0.10
        )

        for bar, value in zip(bars, values):

            ax.text(
                bar.get_x() +
                bar.get_width() / 2,
                value + 2,
                f"{value:.1f}%",
                ha="center",
                color=TEXT,
                fontweight="bold"
            )

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            master=self.chart_frame
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    # ========================================================
    # ABOUT
    # ========================================================

    def _build_about(self):

        self.about_page = self._page_frame()

        self._header(
            self.about_page,
            "Project Profile",
            "Adaptive Multimodal AI for Personalized Healthcare & Wellness Risk Assessment"
        )

        scroll = ctk.CTkScrollableFrame(
            self.about_page,
            fg_color="transparent",
            scrollbar_button_color=CARD_2,
            scrollbar_button_hover_color="#263B58"
        )

        scroll.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=24,
            pady=(0, 24)
        )

        # --------------------------------------------------------
        # Identity
        # --------------------------------------------------------
        identity = self._card(scroll)
        identity.pack(fill="x", padx=8, pady=8)

        ctk.CTkLabel(
            identity,
            text="AegisAI",
            font=ctk.CTkFont(size=38, weight="bold"),
            text_color=ACCENT
        ).pack(pady=(28, 5))

        ctk.CTkLabel(
            identity,
            text=FORMAL_PROJECT_TITLE,
            font=ctk.CTkFont(size=19, weight="bold"),
            text_color=TEXT,
            wraplength=900,
            justify="center"
        ).pack(padx=25)

        ctk.CTkLabel(
            identity,
            text="From Multiple Signals to Smarter Health Decisions.",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=PURPLE
        ).pack(pady=(7, 24))

        # --------------------------------------------------------
        # Student details
        # --------------------------------------------------------
        student = self._card(scroll)
        student.pack(fill="x", padx=8, pady=8)

        self._section_title(
            student,
            "01  Student & Academic Details",
            "Project author and academic background"
        )

        student_text = (
            f"Student Name: {STUDENT_NAME}\n"
            f"Degree: {DEGREE}\n"
            f"University: {UNIVERSITY}\n"
            f"Academic Period: {ACADEMIC_PERIOD}\n"
            f"Academic Year: {ACADEMIC_YEAR}\n"
            f"CGPA: {CGPA}\n"
            f"Primary Specialization: Artificial Intelligence & Machine Learning"
        )

        ctk.CTkLabel(
            student,
            text=student_text,
            font=ctk.CTkFont(size=13),
            text_color=TEXT,
            justify="left",
            anchor="w",
            wraplength=980
        ).pack(fill="x", padx=25, pady=(0, 22))

        # --------------------------------------------------------
        # Domain
        # --------------------------------------------------------
        domain = self._card(scroll)
        domain.pack(fill="x", padx=8, pady=8)

        self._section_title(
            domain,
            "02  Area / Domain of Project",
            "Primary interdisciplinary domain"
        )

        ctk.CTkLabel(
            domain,
            text=PROJECT_DOMAIN,
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=ACCENT,
            wraplength=900
        ).pack(anchor="w", padx=25, pady=(0, 22))

        # --------------------------------------------------------
        # Abstract
        # --------------------------------------------------------
        abstract = self._card(scroll)
        abstract.pack(fill="x", padx=8, pady=8)

        self._section_title(
            abstract,
            "03  Abstract",
            "Project motivation, architecture and intended contribution"
        )

        abstract_text = (
            "The proposed project, “Adaptive Multimodal AI for Personalized "
            "Healthcare & Wellness Risk Assessment,” aims to develop an intelligent "
            "multimodal AI-based system capable of integrating multiple sources of "
            "information to assess a user's general health and wellness condition. "
            "The system analyzes facial expressions, voice characteristics, "
            "user-reported symptoms, physiological measurements, lifestyle patterns, "
            "dietary and hydration information, medical context, and environmental "
            "factors such as air quality and ambient conditions. Machine learning "
            "and deep learning techniques are employed to extract meaningful patterns "
            "from individual modalities, while a multimodal fusion mechanism combines "
            "these signals to generate an overall wellness-risk assessment.\n\n"
            "A key feature of the proposed system is its adaptive and quality-aware "
            "architecture. Since different inputs may vary in reliability, availability, "
            "and quality, the system is designed to consider the contribution of "
            "individual modalities according to the available information. This enables "
            "the system to continue producing useful wellness insights even when some "
            "modalities are incomplete or unavailable.\n\n"
            "The system also incorporates an AI-assisted safety and consistency layer "
            "to identify extreme, inconsistent, or potentially concerning input "
            "conditions before presenting the final assessment. The current prototype "
            "supports a three-level wellness-risk output: Low, Moderate, and High. "
            "Based on the generated assessment, the system is intended to support "
            "personalized and explainable wellness recommendations related to lifestyle, "
            "nutrition, hydration, sleep, physical activity, environmental exposure, "
            "and stress management.\n\n"
            "The proposed system is intended as a research and wellness decision-support "
            "prototype and is not intended to replace professional medical diagnosis, "
            "treatment, or clinical judgment. By integrating physical, emotional, "
            "behavioral, nutritional, and environmental signals, the project aims to "
            "demonstrate how adaptive multimodal AI can provide a more comprehensive "
            "and personalized approach to healthcare and wellness support."
        )

        ctk.CTkLabel(
            abstract,
            text=abstract_text,
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            justify="left",
            anchor="w",
            wraplength=980
        ).pack(fill="x", padx=25, pady=(0, 24))

        # --------------------------------------------------------
        # Problem statement
        # --------------------------------------------------------
        problem = self._card(scroll)
        problem.pack(fill="x", padx=8, pady=8)

        self._section_title(
            problem,
            "04  Problem Statement",
            "Core problem addressed by the proposed system"
        )

        problem_text = (
            "Develop an adaptive AI-based multimodal healthcare and wellness "
            "assistant capable of integrating facial expressions, voice characteristics, "
            "physiological measurements, user-reported symptoms, lifestyle and dietary "
            "information, medical context, and environmental factors to assess general "
            "wellness risk. The system should intelligently combine heterogeneous "
            "information from multiple modalities, account for differences in input "
            "quality and availability, identify potentially concerning combinations "
            "of signals, generate an interpretable wellness-risk assessment, and "
            "provide personalized recommendations while maintaining a clear distinction "
            "between AI-assisted wellness assessment and professional medical diagnosis."
        )

        ctk.CTkLabel(
            problem,
            text=problem_text,
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            justify="left",
            anchor="w",
            wraplength=980
        ).pack(fill="x", padx=25, pady=(0, 24))

        # --------------------------------------------------------
        # Objectives
        # --------------------------------------------------------
        objectives = self._card(scroll)
        objectives.pack(fill="x", padx=8, pady=8)

        self._section_title(
            objectives,
            "05  Project Objectives",
            "Major objectives of the system"
        )

        objective_text = (
            "• Integrate heterogeneous health and wellness signals into one AI workflow.\n"
            "• Extract meaningful features from individual modalities.\n"
            "• Apply adaptive multimodal fusion to combine available information.\n"
            "• Incorporate input-quality awareness and robust handling of incomplete data.\n"
            "• Detect extreme and internally inconsistent assessment inputs.\n"
            "• Generate Low, Moderate, or High wellness-risk estimates.\n"
            "• Provide interpretable reasons behind important risk signals.\n"
            "• Support personalized wellness recommendations.\n"
            "• Demonstrate a scalable research architecture for multimodal AI."
        )

        ctk.CTkLabel(
            objectives,
            text=objective_text,
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            justify="left",
            anchor="w"
        ).pack(fill="x", padx=25, pady=(0, 24))

        # --------------------------------------------------------
        # Multimodal architecture
        # --------------------------------------------------------
        architecture = self._card(scroll)
        architecture.pack(fill="x", padx=8, pady=8)

        self._section_title(
            architecture,
            "06  Multimodal Intelligence Architecture",
            "Information sources used by AegisAI"
        )

        modalities = [
            ("01", "Facial Expression", "Emotional-state indicators"),
            ("02", "Voice & Speech", "Voice stress and tone characteristics"),
            ("03", "Vitals & Blood Pressure", "BP, heart rate, SpO₂, temperature and respiration"),
            ("04", "Symptoms & Current State", "Symptoms, fatigue and current condition"),
            ("05", "Lifestyle & Activity", "Sleep, steps, exercise and sedentary behavior"),
            ("06", "Diet & Hydration", "Water, calories, protein, sugar and fruit/vegetable intake"),
            ("07", "Environment", "AQI, PM2.5, temperature, humidity, noise and UV"),
            ("08", "Medical Context", "Existing conditions, medication and family history"),
        ]

        for number, title, desc in modalities:
            row = ctk.CTkFrame(
                architecture,
                fg_color=CARD_2,
                corner_radius=10
            )
            row.pack(fill="x", padx=18, pady=4)

            ctk.CTkLabel(
                row,
                text=number,
                width=35,
                text_color=ACCENT,
                font=ctk.CTkFont(size=11, weight="bold")
            ).pack(side="left", padx=(12, 6), pady=9)

            ctk.CTkLabel(
                row,
                text=title,
                text_color=TEXT,
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(side="left", padx=5, pady=9)

            ctk.CTkLabel(
                row,
                text=desc,
                text_color=MUTED,
                font=ctk.CTkFont(size=11),
                wraplength=620,
                justify="right"
            ).pack(side="right", padx=12, pady=9)

        ctk.CTkLabel(
            architecture,
            text=(
                "Overall flow: User Data → Multimodal Feature Extraction → "
                "Quality Assessment → Adaptive Fusion → ML Risk Model → "
                "Safety / Consistency Layer → Explainable Wellness-Risk Assessment"
            ),
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=PURPLE,
            wraplength=980,
            justify="left"
        ).pack(fill="x", padx=25, pady=20)

        # --------------------------------------------------------
        # Technical stack
        # --------------------------------------------------------
        stack = self._card(scroll)
        stack.pack(fill="x", padx=8, pady=8)

        self._section_title(
            stack,
            "07  Technical Stack",
            "Technologies used across the project"
        )

        stack_text = (
            "Programming: Python, Java, SQL\n"
            "Machine Learning: Scikit-learn, XGBoost, Random Forest, classification models\n"
            "Deep Learning: TensorFlow, Keras, CNN\n"
            "Computer Vision: OpenCV, facial emotion recognition\n"
            "Data Science: NumPy, Pandas, Matplotlib\n"
            "Application Development: CustomTkinter, Java Swing\n"
            "Database / Storage: MySQL, CSV, Joblib model bundles\n"
            "Core CS Foundations: DSA, OOP, OS, DBMS, Computer Networks, Distributed Computing"
        )

        ctk.CTkLabel(
            stack,
            text=stack_text,
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            justify="left",
            anchor="w"
        ).pack(fill="x", padx=25, pady=(0, 24))

        # --------------------------------------------------------
        # Dataset / model
        # --------------------------------------------------------
        model = self._card(scroll)
        model.pack(fill="x", padx=8, pady=8)

        self._section_title(
            model,
            "08  Dataset & AI Model",
            "Current prototype configuration"
        )

        model_text = (
            f"Dataset: {DATASET_SIZE}\n"
            "Risk classes: Low / Moderate / High\n"
            "Primary model: XGBoost\n"
            "Architecture: ML prediction + deterministic safety / consistency layer\n"
            "Data type: Synthetic research dataset\n"
            "Purpose: Model development, experimentation and prototype evaluation\n"
            "Important: Synthetic-data performance does not establish clinical validity."
        )

        ctk.CTkLabel(
            model,
            text=model_text,
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            justify="left",
            anchor="w"
        ).pack(fill="x", padx=25, pady=(0, 24))

        # --------------------------------------------------------
        # Relevant technical background
        # --------------------------------------------------------
        background = self._card(scroll)
        background.pack(fill="x", padx=8, pady=8)

        self._section_title(
            background,
            "09  Student Technical Profile",
            "Relevant skills and project experience"
        )

        background_text = (
            "Primary Interest: Artificial Intelligence & Machine Learning\n\n"
            "Core technical areas:\n"
            "• Data Structures & Algorithms\n"
            "• Object-Oriented Programming\n"
            "• Operating Systems\n"
            "• Computer Networks\n"
            "• Database Management Systems\n"
            "• Distributed Computing\n"
            "• Machine Learning & Deep Learning\n"
            "• Computer Vision & Multimodal AI\n\n"
            "Relevant projects include AI-powered ransomware detection, fake URL / "
            "phishing detection, stock price prediction, fraud detection, customer "
            "churn prediction, spam SMS detection, movie genre classification, and "
            "AI-based facial emotion recognition."
        )

        ctk.CTkLabel(
            background,
            text=background_text,
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            justify="left",
            anchor="w",
            wraplength=980
        ).pack(fill="x", padx=25, pady=(0, 24))

        # --------------------------------------------------------
        # Intended contribution
        # --------------------------------------------------------
        contribution = self._card(scroll)
        contribution.pack(fill="x", padx=8, pady=8)

        self._section_title(
            contribution,
            "10  Intended Contribution",
            "Research and engineering contribution"
        )

        contribution_text = (
            "The project demonstrates an end-to-end multimodal AI workflow in which "
            "heterogeneous wellness signals are brought together instead of being "
            "considered independently. Its adaptive design emphasizes modality "
            "availability and quality, while the safety and consistency layer adds "
            "a deterministic guard against extreme or internally inconsistent inputs. "
            "The overall objective is to provide a transparent research prototype "
            "for personalized wellness-risk assessment rather than a clinical "
            "diagnostic system."
        )

        ctk.CTkLabel(
            contribution,
            text=contribution_text,
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            justify="left",
            anchor="w",
            wraplength=980
        ).pack(fill="x", padx=25, pady=(0, 24))

        # --------------------------------------------------------
        # Disclaimer
        # --------------------------------------------------------
        warning = self._card(scroll)
        warning.pack(fill="x", padx=8, pady=8)

        ctk.CTkLabel(
            warning,
            text="⚠  Research / Wellness Use",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=WARNING
        ).pack(pady=(18, 6))

        ctk.CTkLabel(
            warning,
            text=(
                "AegisAI is an AI research and wellness decision-support prototype "
                "using synthetic training data. Its output is an estimate for "
                "research and general wellness support. It is not a medical "
                "diagnostic device and should not replace qualified medical advice, "
                "diagnosis, treatment, or emergency care."
            ),
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            wraplength=900,
            justify="center"
        ).pack(padx=25, pady=(0, 22))

    # ========================================================
    # INPUT / PREDICTION
    # ========================================================

    def _number(self, key):

        try:
            value = float(
                self.vars[key].get()
            )

            return value

        except Exception:

            raise ValueError(
                f"Invalid value for {key}"
            )

    def _int(self, key):

        try:
            return int(
                float(
                    self.vars[key].get()
                )
            )

        except Exception:

            raise ValueError(
                f"Invalid value for {key}"
            )

    def _validate_ranges(self):
        """Validate every editable numeric field before any ML call."""
        checks = {
            "Age": (1, 120),
            "Height_cm": (100, 230),
            "Weight_kg": (20, 250),
            "BMI": (10, 70),
            "Stress_Level_0_10": (0, 10),
            "Mood_Score_0_10": (0, 10),
            "Voice_Stress_0_10": (0, 10),
            "Voice_Tone_Score_0_10": (0, 10),
            "Systolic_BP_mmHg": (50, 300),
            "Diastolic_BP_mmHg": (20, 200),
            "Heart_Rate_bpm": (25, 240),
            "SpO2_percent": (50, 100),
            "Body_Temperature_C": (30, 45),
            "Respiratory_Rate_bpm": (4, 70),
            "Fatigue_Score_0_10": (0, 10),
            "Symptom_Count": (0, 20),
            "Sleep_Hours": (0, 24),
            "Daily_Steps": (0, 100000),
            "Exercise_Minutes_Week": (0, 3000),
            "Sedentary_Hours_Day": (0, 24),
            "Water_Liters_Day": (0, 15),
            "Calories_Day": (0, 10000),
            "Protein_g_Day": (0, 500),
            "Sugar_g_Day": (0, 500),
            "Fruit_Veg_Servings_Day": (0, 30),
            "AQI": (0, 1000),
            "PM2_5_ug_m3": (0, 1000),
            "Temperature_Environment_C": (-30, 60),
            "Humidity_percent": (0, 100),
            "Noise_dB": (0, 150),
            "UV_Index": (0, 20),
        }

        for key, (low, high) in checks.items():
            value = self._number(key)
            if not math.isfinite(value):
                raise ValueError(f"{key} must be a finite numeric value.")
            if not low <= value <= high:
                raise ValueError(
                    f"{key} must be between {low} and {high}."
                )

    def _derive_wellness_score(self, data):
        """
        Reproducible fallback wellness score used only when an imported
        model explicitly expects Wellness_Score_0_100.

        This is a compatibility feature, not a medical score and not a
        substitute for retraining a model without target-adjacent leakage.
        """
        def clamp(x, lo=0.0, hi=100.0):
            return max(lo, min(hi, float(x)))

        score = 75.0
        score += (data["Mood_Score_0_10"] - 5.0) * 2.2
        score -= data["Stress_Level_0_10"] * 1.8
        score -= data["Voice_Stress_0_10"] * 0.9
        score += (data["Voice_Tone_Score_0_10"] - 5.0) * 0.8
        score += (data["Sleep_Hours"] - 7.0) * 2.0
        score += min(data["Daily_Steps"], 12000) / 12000 * 4.0
        score += min(data["Exercise_Minutes_Week"], 300) / 300 * 3.0
        score -= max(0, data["Sedentary_Hours_Day"] - 8) * 1.2
        score += min(data["Water_Liters_Day"], 4) * 1.0
        score -= max(0, data["Sugar_g_Day"] - 50) / 50 * 1.5
        score += min(data["Fruit_Veg_Servings_Day"], 8) * 0.7

        # Physiological/environmental penalties
        if data["Systolic_BP_mmHg"] >= 140: score -= 7
        elif data["Systolic_BP_mmHg"] >= 130: score -= 3

        if data["Diastolic_BP_mmHg"] >= 90: score -= 6
        elif data["Diastolic_BP_mmHg"] >= 80: score -= 2

        if data["SpO2_percent"] < 95: score -= (95 - data["SpO2_percent"]) * 2.5
        if data["Body_Temperature_C"] >= 38: score -= (data["Body_Temperature_C"] - 37) * 5
        if data["AQI"] >= 150: score -= min(15, (data["AQI"] - 150) / 30)
        if data["PM2_5_ug_m3"] >= 55: score -= min(12, (data["PM2_5_ug_m3"] - 55) / 10)
        if data["BMI"] >= 30: score -= min(8, (data["BMI"] - 30) * 0.7)
        score -= min(8, data["Fatigue_Score_0_10"] * 0.7)
        score -= min(10, data["Symptom_Count"] * 1.2)

        return round(clamp(score), 2)

    def _model_expected_features(self):
        """Return feature names expected by the loaded model, when discoverable."""
        bundle = self.model_bundle
        if not isinstance(bundle, dict):
            return None

        if isinstance(bundle.get("features"), (list, tuple, np.ndarray)):
            return [str(x) for x in bundle["features"]]

        pipeline = bundle.get("pipeline")
        if pipeline is not None:
            names = getattr(pipeline, "feature_names_in_", None)
            if names is not None:
                return [str(x) for x in names]

            # ColumnTransformer / pipeline compatibility.
            pre = getattr(pipeline, "named_steps", {}).get("preprocessor")
            names = getattr(pre, "feature_names_in_", None)
            if names is not None:
                return [str(x) for x in names]

        return None

    def _collect_input(self):

        self._validate_ranges()

        data = {

            "Age":
                self._int("Age"),

            "Sex":
                self.vars["Sex"].get(),

            "Height_cm":
                self._number("Height_cm"),

            "Weight_kg":
                self._number("Weight_kg"),

            "BMI":
                self._number("BMI"),

            "Existing_Condition":
                self.vars["Existing_Condition"].get(),

            "Medication_Status":
                self.vars["Medication_Status"].get(),

            "Family_History":
                self.vars["Family_History"].get(),

            "Facial_Expression":
                self.vars["Facial_Expression"].get(),

            "Stress_Level_0_10":
                self._number("Stress_Level_0_10"),

            "Mood_Score_0_10":
                self._number("Mood_Score_0_10"),

            "Voice_Stress_0_10":
                self._number("Voice_Stress_0_10"),

            "Voice_Tone_Score_0_10":
                self._number("Voice_Tone_Score_0_10"),

            "Systolic_BP_mmHg":
                self._number("Systolic_BP_mmHg"),

            "Diastolic_BP_mmHg":
                self._number("Diastolic_BP_mmHg"),

            "Heart_Rate_bpm":
                self._number("Heart_Rate_bpm"),

            "SpO2_percent":
                self._number("SpO2_percent"),

            "Body_Temperature_C":
                self._number("Body_Temperature_C"),

            "Respiratory_Rate_bpm":
                self._number("Respiratory_Rate_bpm"),

            "Fatigue_Score_0_10":
                self._number("Fatigue_Score_0_10"),

            "Symptom_Count":
                self._int("Symptom_Count"),

            "Primary_Symptom":
                self.vars["Primary_Symptom"].get(),

            "Sleep_Hours":
                self._number("Sleep_Hours"),

            "Daily_Steps":
                self._int("Daily_Steps"),

            "Exercise_Minutes_Week":
                self._int("Exercise_Minutes_Week"),

            "Sedentary_Hours_Day":
                self._number("Sedentary_Hours_Day"),

            "Smoking_Status":
                self.vars["Smoking_Status"].get(),

            "Alcohol_Frequency":
                self.vars["Alcohol_Frequency"].get(),

            "Water_Liters_Day":
                self._number("Water_Liters_Day"),

            "Calories_Day":
                self._int("Calories_Day"),

            "Protein_g_Day":
                self._number("Protein_g_Day"),

            "Sugar_g_Day":
                self._number("Sugar_g_Day"),

            "Fruit_Veg_Servings_Day":
                self._number("Fruit_Veg_Servings_Day"),

            "AQI":
                self._int("AQI"),

            "PM2_5_ug_m3":
                self._number("PM2_5_ug_m3"),

            "Temperature_Environment_C":
                self._number(
                    "Temperature_Environment_C"
                ),

            "Humidity_percent":
                self._number("Humidity_percent"),

            "Noise_dB":
                self._number("Noise_dB"),

            "UV_Index":
                self._number("UV_Index")
        }

        return data


    # ========================================================
    # SAFETY / CORNER-CASE ENGINE
    # ========================================================

    def _validate_cross_field_consistency(self, data):
        """
        Reject impossible/internal-contradiction inputs while allowing
        dangerous-but-plausible inputs to reach the safety layer.
        """
        errors = []

        sbp, dbp = data["Systolic_BP_mmHg"], data["Diastolic_BP_mmHg"]
        height, weight = data["Height_cm"], data["Weight_kg"]
        bmi = data["BMI"]

        if sbp < dbp:
            errors.append("Systolic BP cannot be lower than diastolic BP.")
        if sbp - dbp < 10:
            errors.append("Systolic and diastolic BP are unrealistically close; recheck the measurement.")

        calculated_bmi = weight / ((height / 100.0) ** 2)
        if abs(calculated_bmi - bmi) > max(2.0, calculated_bmi * 0.12):
            errors.append(
                f"BMI ({bmi:.1f}) does not match height/weight "
                f"(calculated ≈ {calculated_bmi:.1f})."
            )

        if data["SpO2_percent"] > 100:
            errors.append("SpO₂ cannot be above 100%.")

        if data["Body_Temperature_C"] < 32 or data["Body_Temperature_C"] > 45:
            errors.append("Body temperature must be between 32°C and 45°C.")

        if data["Temperature_Environment_C"] < -30 or data["Temperature_Environment_C"] > 60:
            errors.append("Ambient temperature must be between -30°C and 60°C.")

        if data["Sleep_Hours"] + data["Sedentary_Hours_Day"] > 24:
            errors.append("Sleep hours + sedentary hours/day cannot exceed 24 hours.")

        if data["Exercise_Minutes_Week"] > 1000 and data["Daily_Steps"] < 500:
            errors.append("Exercise minutes and daily steps appear internally inconsistent.")

        if data["Symptom_Count"] == 0 and data["Primary_Symptom"] != "None":
            errors.append("Primary symptom is selected but symptom count is 0.")

        if data["Symptom_Count"] > 0 and data["Primary_Symptom"] == "None":
            errors.append("Symptom count is above 0, so select a primary symptom.")

        if data["Existing_Condition"] == "None" and data["Medication_Status"] != "None":
            errors.append("Medication cannot be marked active when no existing condition is selected.")

        if data["Age"] < 13:
            errors.append("This version is designed for an adult-oriented research profile (age ≥ 13).")

        # Environmental consistency checks.
        if data["PM2_5_ug_m3"] > 500 and data["AQI"] < 100:
            errors.append("AQI and PM2.5 appear inconsistent; recheck the environmental reading.")

        if data["UV_Index"] > 15 and data["Temperature_Environment_C"] < -10:
            errors.append("UV index and ambient temperature appear inconsistent; recheck live/manual values.")

        return errors

    def _safety_risk_override(self, data):
        """
        Deterministic safety layer for dangerous-but-plausible patterns.

        The ML result remains the normal classifier output. This layer is
        intentionally conservative and only upgrades to High when a strong
        single signal or multiple independent signals are present.
        """
        reasons = []
        score = 0

        sbp, dbp = data["Systolic_BP_mmHg"], data["Diastolic_BP_mmHg"]
        spo2 = data["SpO2_percent"]
        temp = data["Body_Temperature_C"]
        hr = data["Heart_Rate_bpm"]
        rr = data["Respiratory_Rate_bpm"]
        water = data["Water_Liters_Day"]
        sugar = data["Sugar_g_Day"]
        calories = data["Calories_Day"]
        stress = data["Stress_Level_0_10"]
        fatigue = data["Fatigue_Score_0_10"]
        symptoms = data["Symptom_Count"]
        primary = data["Primary_Symptom"]
        sleep = data["Sleep_Hours"]
        aqi, pm25 = data["AQI"], data["PM2_5_ug_m3"]
        bmi = data["BMI"]
        noise = data["Noise_dB"]

        # Critical measurements.
        if sbp >= 180 or dbp >= 120:
            score += 6; reasons.append(f"Very high blood pressure ({sbp:.0f}/{dbp:.0f} mmHg)")
        elif sbp >= 160 or dbp >= 100:
            score += 3; reasons.append(f"Markedly elevated blood pressure ({sbp:.0f}/{dbp:.0f} mmHg)")

        if spo2 <= 90:
            score += 6; reasons.append(f"Very low SpO₂ ({spo2:.1f}%)")
        elif spo2 <= 94:
            score += 3; reasons.append(f"Reduced SpO₂ ({spo2:.1f}%)")

        if temp >= 40:
            score += 5; reasons.append(f"Very high body temperature ({temp:.1f}°C)")
        elif temp >= 39:
            score += 3; reasons.append(f"High body temperature ({temp:.1f}°C)")

        if hr >= 140 or hr <= 40:
            score += 4; reasons.append(f"Extreme heart rate ({hr:.0f} bpm)")
        elif hr >= 120 or hr <= 45:
            score += 2; reasons.append(f"Abnormal heart rate ({hr:.0f} bpm)")

        if rr >= 30 or rr <= 8:
            score += 4; reasons.append(f"Extreme respiratory rate ({rr:.0f}/min)")
        elif rr >= 24 or rr <= 10:
            score += 2; reasons.append(f"Abnormal respiratory rate ({rr:.0f}/min)")

        # Symptoms + vitals.
        if primary == "Breathlessness":
            score += 3; reasons.append("Breathlessness reported.")
        if primary == "Dizziness":
            score += 1; reasons.append("Dizziness reported.")
        if symptoms >= 5:
            score += 2; reasons.append(f"High symptom count ({symptoms})")
        if temp >= 39 and symptoms >= 2:
            score += 3; reasons.append("High temperature combined with multiple symptoms.")
        if spo2 <= 94 and rr >= 24:
            score += 4; reasons.append("Reduced SpO₂ combined with elevated respiratory rate.")
        if sbp >= 160 and (primary in {"Dizziness", "Breathlessness"} or symptoms >= 3):
            score += 3; reasons.append("Markedly elevated BP with concerning symptoms.")

        # Lifestyle/nutrition/environment.
        if water <= 0.2:
            score += 4; reasons.append(f"Extremely low water intake ({water:.2f} L/day)")
        elif water <= 0.75:
            score += 2; reasons.append(f"Very low water intake ({water:.2f} L/day)")

        if sugar >= 300:
            score += 4; reasons.append(f"Extremely high sugar intake ({sugar:.0f} g/day)")
        elif sugar >= 200:
            score += 2; reasons.append(f"Very high sugar intake ({sugar:.0f} g/day)")

        if calories <= 700:
            score += 3; reasons.append(f"Extremely low calorie intake ({calories:.0f} kcal/day)")
        elif calories >= 5000:
            score += 2; reasons.append(f"Extremely high calorie intake ({calories:.0f} kcal/day)")

        if stress >= 9:
            score += 2; reasons.append(f"Very high stress ({stress:.1f}/10)")
        if fatigue >= 8:
            score += 1; reasons.append(f"Severe fatigue ({fatigue:.1f}/10)")
        if sleep < 4:
            score += 2; reasons.append(f"Very low sleep ({sleep:.1f} hours)")

        if aqi >= 300 or pm25 >= 150:
            score += 3; reasons.append(f"Severe air pollution (AQI {aqi:.0f}, PM2.5 {pm25:.1f})")
        elif aqi >= 200 or pm25 >= 100:
            score += 2; reasons.append(f"High air pollution (AQI {aqi:.0f}, PM2.5 {pm25:.1f})")

        if bmi >= 40:
            score += 2; reasons.append(f"Very high BMI ({bmi:.1f})")
        if noise >= 100:
            score += 1; reasons.append(f"Very high environmental noise ({noise:.0f} dB)")
        if data["UV_Index"] >= 11:
            score += 1; reasons.append(f"Very high UV index ({data['UV_Index']:.1f})")

        # Strong combinations.
        if water <= 0.75 and temp >= 38:
            score += 2; reasons.append("Low hydration combined with elevated body temperature.")
        if sugar >= 200 and bmi >= 30:
            score += 1; reasons.append("High sugar intake combined with elevated BMI.")
        if stress >= 8 and sleep < 5 and fatigue >= 7:
            score += 2; reasons.append("High stress + poor sleep + severe fatigue pattern.")

        return ("High" if score >= 4 else None), reasons, score

    def _normalize_model_bundle(self):
        """
        Support both model bundle formats:
        1) pipeline + label_encoder
        2) preprocessor + model + features + label_encoder
        """
        bundle = self.model_bundle

        if not isinstance(bundle, dict):
            raise ValueError("Invalid AegisAI model bundle format.")

        if "pipeline" in bundle:
            pipeline = bundle["pipeline"]
            encoder = bundle.get("label_encoder")
            return pipeline, encoder, None

        required = ["preprocessor", "model", "label_encoder", "features"]
        missing = [k for k in required if k not in bundle]

        if missing:
            raise ValueError(
                "Model bundle is missing required keys: "
                + ", ".join(missing)
            )

        return (
            None,
            bundle["label_encoder"],
            {
                "preprocessor": bundle["preprocessor"],
                "model": bundle["model"],
                "features": bundle["features"],
            },
        )

    def _model_predict(self, data):
        """Run the loaded model with schema-aware feature preparation."""
        pipeline, encoder, legacy = self._normalize_model_bundle()
        input_df = pd.DataFrame([data])

        expected = self._model_expected_features()
        if expected:
            # Compatibility for models trained with Wellness_Score_0_100.
            # Prefer retraining without this field if possible because it is
            # derived/target-adjacent and can cause leakage.
            if "Wellness_Score_0_100" in expected and "Wellness_Score_0_100" not in input_df:
                input_df["Wellness_Score_0_100"] = self._derive_wellness_score(data)

            missing = [f for f in expected if f not in input_df.columns]
            if missing:
                raise ValueError(
                    "Loaded model expects features not supplied by the GUI: "
                    + ", ".join(missing)
                    + ". Retrain the model using the same feature schema."
                )

        if pipeline is not None:
            prediction_encoded = pipeline.predict(input_df)[0]
            probabilities = (
                pipeline.predict_proba(input_df)[0]
                if hasattr(pipeline, "predict_proba") else None
            )
            classes = getattr(pipeline, "classes_", None)
        else:
            preprocessor = legacy["preprocessor"]
            model = legacy["model"]
            features = legacy["features"]
            input_df = input_df[features]
            processed = preprocessor.transform(input_df)
            prediction_encoded = model.predict(processed)[0]
            probabilities = (
                model.predict_proba(processed)[0]
                if hasattr(model, "predict_proba") else None
            )
            classes = getattr(model, "classes_", None)

        if encoder is not None:
            try:
                risk = str(encoder.inverse_transform([prediction_encoded])[0])
            except Exception:
                risk = str(prediction_encoded)
        else:
            risk = str(prediction_encoded)

        # Normalize class names and probabilities.
        probability_dict = {}
        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=float)
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
            probabilities = np.clip(probabilities, 0.0, None)
            total = float(probabilities.sum())
            if total > 0:
                probabilities /= total

            if encoder is not None:
                try:
                    labels = encoder.inverse_transform(np.arange(len(probabilities)))
                except Exception:
                    labels = classes if classes is not None else np.arange(len(probabilities))
            else:
                labels = classes if classes is not None else np.arange(len(probabilities))

            probability_dict = {
                str(label): float(prob)
                for label, prob in zip(labels, probabilities)
            }

        # Always provide the three application classes when probabilities exist.
        if probability_dict:
            for label in ("Low", "Moderate", "High"):
                probability_dict.setdefault(label, 0.0)
            total = sum(max(0.0, float(v)) for v in probability_dict.values())
            if total > 0:
                probability_dict = {
                    k: max(0.0, float(v)) / total
                    for k, v in probability_dict.items()
                }

        if risk not in {"Low", "Moderate", "High"}:
            # Numeric encoded models sometimes expose 0/1/2 labels.
            mapping = {"0": "Low", "1": "Moderate", "2": "High"}
            risk = mapping.get(risk, risk)

        if risk not in {"Low", "Moderate", "High"}:
            raise ValueError(f"Model returned unsupported risk class: {risk}")

        # If predict_proba is unavailable, expose a truthful one-hot result.
        if not probability_dict:
            probability_dict = {"Low": 0.0, "Moderate": 0.0, "High": 0.0}
            probability_dict[risk] = 1.0

        return risk, probability_dict


    def run_prediction(self):
        if not self.model_loaded:
            messagebox.showerror(
                "Model Not Found",
                f"{MODEL_FILENAME} was not found in the application folder."
            )
            return

        try:
            data = self._collect_input()

            consistency_errors = self._validate_cross_field_consistency(data)
            if consistency_errors:
                messagebox.showerror(
                    "Invalid / Inconsistent Input",
                    "Please correct the following:\n\n• "
                    + "\n• ".join(consistency_errors)
                )
                return

            ml_risk, probability_dict = self._model_predict(data)
            forced_risk, safety_reasons, safety_score = self._safety_risk_override(data)
            final_risk = forced_risk if forced_risk is not None else ml_risk

            # Keep displayed probabilities internally consistent with the final
            # safety-layer result. These are UI probabilities, not calibrated
            # medical probabilities.
            if forced_risk == "High":
                high_floor = 0.90 if safety_score >= 7 else 0.80
                high = max(float(probability_dict.get("High", 0.0)), high_floor)
                high = min(high, 0.99)
                remaining = 1.0 - high
                others = [k for k in ("Low", "Moderate") if k in probability_dict]
                old_other = sum(max(0.0, float(probability_dict.get(k, 0.0))) for k in others)

                if old_other > 0:
                    for k in others:
                        probability_dict[k] = (
                            max(0.0, float(probability_dict.get(k, 0.0)))
                            / old_other * remaining
                        )
                else:
                    for k in others:
                        probability_dict[k] = remaining / max(1, len(others))
                probability_dict["High"] = high

            # Emergency escalation is independent of the ML class.
            emergency_flags = self._emergency_flags(data)

            self.current_result = {
                "risk": final_risk,
                "ml_risk": ml_risk,
                "probabilities": probability_dict,
                "safety_override": forced_risk is not None,
                "safety_score": safety_score,
                "safety_reasons": safety_reasons,
                "emergency_flags": emergency_flags,
                "input_quality": "Validated",
            }

            self._update_result(
                final_risk,
                probability_dict,
                safety_override=(forced_risk is not None),
                safety_reasons=safety_reasons,
                ml_risk=ml_risk,
                emergency_flags=emergency_flags,
            )
            self.show_page("results")

        except Exception as e:
            messagebox.showerror("Assessment Error", str(e))

    def _emergency_flags(self, data):
        """Hard escalation flags; never claim a diagnosis."""
        flags = []

        if data["Systolic_BP_mmHg"] >= 180 or data["Diastolic_BP_mmHg"] >= 120:
            flags.append("Very high BP reading — repeat correctly after rest and seek urgent medical assessment, especially with symptoms.")

        if data["SpO2_percent"] <= 90:
            flags.append("SpO₂ is very low — urgent medical assessment is appropriate, particularly with breathing difficulty.")

        if data["Primary_Symptom"] == "Breathlessness" and data["SpO2_percent"] <= 94:
            flags.append("Breathlessness with reduced SpO₂ requires prompt medical assessment.")

        if data["Body_Temperature_C"] >= 40:
            flags.append("Very high body temperature — urgent medical assessment is appropriate.")

        if data["Heart_Rate_bpm"] >= 150 or data["Heart_Rate_bpm"] <= 35:
            flags.append("Extreme heart-rate reading — verify the measurement and seek urgent care if persistent or symptomatic.")

        if data["Respiratory_Rate_bpm"] >= 35 or data["Respiratory_Rate_bpm"] <= 6:
            flags.append("Extreme respiratory-rate reading — seek urgent assessment if confirmed.")

        if data["Primary_Symptom"] == "Breathlessness" and data["Respiratory_Rate_bpm"] >= 30:
            flags.append("Breathlessness with markedly elevated respiratory rate — urgent assessment.")

        return flags

    def _update_result(
        self,
        risk,
        probabilities,
        safety_override=False,
        safety_reasons=None,
        ml_risk=None,
        emergency_flags=None
    ):

        color_map = {
            "Low": SUCCESS,
            "Moderate": WARNING,
            "High": DANGER
        }

        color = color_map.get(
            risk,
            ACCENT
        )

        self.result_title.configure(
            text="AI Wellness Assessment"
        )

        self.result_risk.configure(
            text=risk.upper(),
            text_color=color
        )

        confidence = probabilities.get(
            risk,
            0
        ) * 100

        if safety_override:
            self.result_confidence.configure(
                text=(
                    f"Safety override: HIGH RISK  •  "
                    f"Model probability: {confidence:.1f}%"
                ),
                text_color=DANGER
            )
        else:
            self.result_confidence.configure(
                text=(
                    f"Model probability: "
                    f"{confidence:.1f}%"
                ),
                text_color=color
            )

        messages = {
            "Low": (
                "The model estimates lower wellness risk based on "
                "the submitted feature profile."
            ),
            "Moderate": (
                "The model estimates moderate wellness risk. "
                "Review lifestyle and health indicators and consider "
                "professional advice when appropriate."
            ),
            "High": (
                "The assessment detected a higher-risk feature profile. "
                "Review the reported indicators and seek appropriate "
                "professional medical advice when symptoms or severe "
                "measurements are present."
            )
        }

        message = messages.get(risk, "")

        if safety_override:
            reason_text = "\n\nSafety checks triggered:\n• " + "\n• ".join(
                safety_reasons[:6]
            )
            message += reason_text

            if ml_risk and ml_risk != "High":
                message += (
                    f"\n\nML model alone predicted: {ml_risk}. "
                    "The safety layer upgraded the final result because "
                    "extreme input conditions were detected."
                )

        if emergency_flags:
            message += (
                "\n\nURGENT ESCALATION — DO NOT WAIT FOR THE AI LABEL:\n• "
                + "\n• ".join(emergency_flags[:5])
                + "\n\nIf severe symptoms are present, contact local emergency services (112 in India)."
            )

        self.result_message.configure(
            text=message
        )

        self._draw_chart(
            probabilities
        )

    def _build_complete_report(self, data, risk, probabilities, safety_reasons):
        """Build a practical wellness report with self-care, first-aid and escalation guidance.

        This is decision-support content, not a medical diagnosis or treatment plan.
        """
        lines = []
        lines.append("AEGISAI — COMPLETE HEALTH & WELLNESS REPORT")
        lines.append("=" * 62)
        lines.append("")
        lines.append(f"Overall AI wellness risk: {risk.upper()}")
        if probabilities:
            lines.append("Model probabilities:")
            for label, prob in probabilities.items():
                lines.append(f"  • {label}: {float(prob) * 100:.1f}%")
        lines.append("")
        lines.append("IMPORTANT")
        lines.append("AegisAI is a research/wellness decision-support prototype. It does")
        lines.append("not diagnose disease and must not replace a qualified clinician or")
        lines.append("emergency care. Urgency is determined by symptoms and measurements,")
        lines.append("not by the AI risk label alone.")
        lines.append("")

        # -------------------------
        # 1. Key measurements
        # -------------------------
        lines.append("1. KEY ASSESSMENT FINDINGS")
        lines.append("-" * 62)
        lines.append(f"  • Blood pressure: {data['Systolic_BP_mmHg']:.0f}/{data['Diastolic_BP_mmHg']:.0f} mmHg")
        lines.append(f"  • SpO₂: {data['SpO2_percent']:.1f}%")
        lines.append(f"  • Heart rate: {data['Heart_Rate_bpm']:.0f} bpm")
        lines.append(f"  • Body temperature: {data['Body_Temperature_C']:.1f} °C")
        lines.append(f"  • Respiratory rate: {data['Respiratory_Rate_bpm']:.0f}/min")
        lines.append(f"  • BMI: {data['BMI']:.1f}")
        lines.append(f"  • Sleep: {data['Sleep_Hours']:.1f} h/day")
        lines.append(f"  • Water: {data['Water_Liters_Day']:.1f} L/day")
        lines.append(f"  • Stress: {data['Stress_Level_0_10']:.0f}/10")
        lines.append(f"  • Fatigue: {data['Fatigue_Score_0_10']:.0f}/10")
        lines.append(f"  • Symptoms reported: {data['Symptom_Count']:.0f}")
        lines.append(f"  • AQI: {data['AQI']:.0f} | PM2.5: {data['PM2_5_ug_m3']:.1f} µg/m³ | UV: {data['UV_Index']:.1f}")
        lines.append("")

        # -------------------------
        # 2. Risk drivers
        # -------------------------
        lines.append("2. IMPORTANT RISK / SAFETY FLAGS")
        lines.append("-" * 62)
        if safety_reasons:
            for reason in safety_reasons[:10]:
                lines.append(f"  • {reason}")
        else:
            lines.append("  • No deterministic extreme-value safety override was triggered.")
            lines.append("  • Continue monitoring symptoms and routine health indicators.")
        lines.append("")

        # -------------------------
        # 3. Immediate self-care
        # -------------------------
        lines.append("3. BASIC WELLNESS / SELF-CARE")
        lines.append("-" * 62)
        lines.append("  • Rest and avoid strenuous activity if you feel unwell, dizzy or unusually weak.")
        lines.append("  • Maintain normal hydration unless a clinician has restricted your fluid intake.")
        lines.append("  • Prefer balanced meals with vegetables/fruit, adequate protein and less added sugar.")
        lines.append("  • Aim for a consistent sleep schedule and adequate sleep.")
        lines.append("  • Avoid smoking and limit alcohol, especially when symptoms are present.")
        lines.append("  • Recheck abnormal vital signs with a reliable device after resting, when appropriate.")
        lines.append("  • Do not start, stop or change prescription medicines based only on this report.")
        lines.append("")

        # -------------------------
        # 4. First aid
        # -------------------------
        lines.append("4. BASIC FIRST-AID — ONLY WHILE ARRANGING APPROPRIATE CARE")
        lines.append("-" * 62)
        lines.append("  • Severe bleeding: apply firm direct pressure with clean cloth/gauze and call emergency services.")
        lines.append("  • Minor burn: stop the heat source, cool with clean cool running water for at least 10 minutes, and loosely cover.")
        lines.append("  • Choking: if the person cannot cough, speak or breathe, call emergency services and use trained choking first aid.")
        lines.append("  • Fainting: protect from injury, check responsiveness/breathing, and seek emergency help if they do not recover normally.")
        lines.append("  • Suspected stroke: remember FAST — Face, Arm, Speech, Time; seek emergency help immediately.")
        lines.append("  • Suspected heart attack: keep the person resting and call emergency medical help; do not drive yourself.")
        lines.append("  • Do not put butter, toothpaste or other household substances on burns; do not break burn blisters.")
        lines.append("")

        # -------------------------
        # 5. When / whom to consult
        # -------------------------
        lines.append("5. WHEN AND WHOM TO CONSULT")
        lines.append("-" * 62)
        lines.append("  • General Physician / Internal Medicine: persistent symptoms, repeated abnormal vitals, fever, fatigue, dizziness, headache, cough, nausea or general health concerns.")
        lines.append("  • Cardiologist: repeated high blood pressure, concerning heart-rate findings, recurrent chest discomfort or clinician-recommended cardiac evaluation.")
        lines.append("  • Pulmonologist / Emergency Department: significant or persistent breathing difficulty, low oxygen readings, or worsening respiratory symptoms.")
        lines.append("  • Neurologist / Emergency Department: sudden facial weakness, one-sided weakness/numbness, speech difficulty, sudden severe neurological symptoms.")
        lines.append("  • Mental-health professional: persistent severe stress, anxiety, mood problems or functional impairment; emergency help for immediate self-harm risk.")
        lines.append("  • Dietitian / Nutritionist: persistent diet, hydration, weight or nutrition concerns, especially when medical conditions are present.")
        lines.append("")

        # -------------------------
        # 6. Emergency escalation
        # -------------------------
        lines.append("6. EMERGENCY — DO NOT WAIT FOR THE AI RESULT")
        lines.append("-" * 62)
        lines.append("Seek emergency medical help immediately for symptoms such as:")
        lines.append("  • severe difficulty breathing, gasping, choking or inability to speak normally")
        lines.append("  • chest pressure/pain, especially if it spreads to the arm, back, neck or jaw")
        lines.append("  • sudden stroke signs: face drooping, arm weakness or speech difficulty")
        lines.append("  • loss of consciousness, seizure, severe confusion or inability to wake normally")
        lines.append("  • major uncontrolled bleeding, severe injury or serious burns")
        lines.append("  • blue/grey lips or skin, or severe deterioration")
        lines.append("  • a reading such as BP ≥180/120 mmHg together with concerning symptoms requires emergency evaluation")
        lines.append("")
        lines.append("In India, call 112 for a medical emergency / ambulance assistance.")
        lines.append("")

        lines.append("7. FOLLOW-UP")
        lines.append("-" * 62)
        if risk == "High":
            lines.append("  • Because the overall assessment is HIGH, arrange professional medical evaluation based on the symptoms and abnormal measurements.")
        elif risk == "Moderate":
            lines.append("  • Review the modifiable findings and consider a routine clinician appointment if abnormalities persist or symptoms continue.")
        else:
            lines.append("  • Continue healthy routines and routine preventive healthcare; seek care if new or worsening symptoms appear.")
        lines.append("")
        lines.append("Report generated by AegisAI. This report is educational decision-support content, not a diagnosis.")
        return "\n".join(lines)

    def show_complete_report(self):
        """Display the complete report in a scrollable window."""
        if not self.current_result:
            messagebox.showinfo("No Assessment", "Run an assessment first to generate the complete report.")
            return

        try:
            data = self._collect_input()
        except Exception:
            data = {}

        report = self._build_complete_report(
            data,
            self.current_result.get("risk", "Unknown"),
            self.current_result.get("probabilities", {}),
            self.current_result.get("safety_reasons", [])
        )

        window = ctk.CTkToplevel(self)
        window.title("AegisAI — Complete Health & Wellness Report")
        window.geometry("900x760")
        window.minsize(720, 600)
        window.configure(fg_color=BG)
        window.transient(self)

        header = ctk.CTkLabel(
            window,
            text="Complete Health & Wellness Report",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=TEXT
        )
        header.pack(anchor="w", padx=25, pady=(20, 4))

        sub = ctk.CTkLabel(
            window,
            text="AI decision-support summary • self-care • first aid • doctor escalation",
            font=ctk.CTkFont(size=11),
            text_color=MUTED
        )
        sub.pack(anchor="w", padx=25, pady=(0, 12))

        box = ctk.CTkTextbox(
            window,
            fg_color=CARD,
            border_width=1,
            border_color=BORDER,
            text_color=TEXT,
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=12,
            wrap="word"
        )
        box.pack(fill="both", expand=True, padx=25, pady=(0, 15))
        box.insert("1.0", report)
        box.configure(state="disabled")

        ctk.CTkButton(
            window,
            text="Close",
            height=40,
            width=130,
            fg_color=CARD_2,
            hover_color="#24334B",
            text_color=TEXT,
            command=window.destroy
        ).pack(pady=(0, 18))

    def reset_form(self):

        self.current_result = None

        defaults = {

            "Age": "22",
            "Height_cm": "175",
            "Weight_kg": "68",
            "BMI": "22.2",

            "Stress_Level_0_10": "3",
            "Mood_Score_0_10": "8",
            "Voice_Stress_0_10": "3",
            "Voice_Tone_Score_0_10": "8",

            "Systolic_BP_mmHg": "118",
            "Diastolic_BP_mmHg": "76",
            "Heart_Rate_bpm": "68",
            "SpO2_percent": "98",
            "Body_Temperature_C": "36.7",
            "Respiratory_Rate_bpm": "15",

            "Fatigue_Score_0_10": "2",
            "Symptom_Count": "0",

            "Sleep_Hours": "7.5",
            "Daily_Steps": "8500",
            "Exercise_Minutes_Week": "180",
            "Sedentary_Hours_Day": "6",

            "Water_Liters_Day": "2.5",
            "Calories_Day": "2200",
            "Protein_g_Day": "80",
            "Sugar_g_Day": "40",
            "Fruit_Veg_Servings_Day": "5",

            "AQI": "80",
            "PM2_5_ug_m3": "30",
            "Temperature_Environment_C": "27",
            "Humidity_percent": "55",
            "Noise_dB": "50",
            "UV_Index": "5"
        }

        for key, value in defaults.items():

            if key in self.vars:
                self.vars[key].set(value)

        select_defaults = {

            "Sex": "Male",
            "Existing_Condition": "None",
            "Medication_Status": "None",
            "Family_History": "None",
            "Facial_Expression": "Neutral",
            "Primary_Symptom": "None",
            "Smoking_Status": "Never",
            "Alcohol_Frequency": "None"
        }

        for key, value in select_defaults.items():

            if key in self.vars:
                self.vars[key].set(value)

        # Re-fetch current environmental conditions after a form reset.
        self.after(150, self.fetch_live_environment)

    # ========================================================
    # RUN
    # ========================================================


if __name__ == "__main__":

    app = AegisAIApp()

    app.mainloop()
