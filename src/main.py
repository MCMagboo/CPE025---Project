"""OrthoScan AI - clinical desktop UI shell.

A clinician signs in first (see login.py), then the three-column workspace
opens. An X-ray comes in from a PNG/JPEG/DICOM file (see ingestion.py) or a
webcam photo of a print (see camera.py), and is analyzed automatically
(see inference.py): FDI tooth numbering, impacted teeth, and an experimental
spacing estimate. Results stay AI suggestions until the clinician approves.
Export is still a stub, so nothing is saved.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import tkinter
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageOps

from analysis import LOWER_ARCH, UPPER_ARCH
from auth import ClinicianStore
from camera import CameraStream, camera_name, list_cameras
from export_dialog import ExportDialog
from inference import AnalysisUnavailable, Analyzer
from ingestion import FILE_TYPES, XrayLoadError, load_xray
from login import LoginView
from overlay import draw_analysis_overlay, draw_dashed_rect
from report import ReportData, save_report_pdf
from theme import (
    ACCENT,
    ACCENT_HOVER,
    BG_CARD,
    BG_MAIN,
    BG_PLACEHOLDER,
    BG_SIDEBAR,
    BG_SURFACE,
    BORDER,
    DANGER,
    FONT_FAMILY,
    PENDING,
    SUCCESS,
    SUCCESS_HOVER,
    TEXT_MUTED,
    TEXT_PRIMARY,
    CROWDING_BOX,
    SPACING_BOX,
    TOOTH_BOX,
)

# (finding key, title, subtitle, colour of this finding's boxes on the X-ray).
# Clicking a card highlights its finding's boxes; the others fade to overlay.DIMMED_ALPHA.
ASSESSMENT_CARDS = [
    ("fdi", "FDI Tooth Detection", "Tooth numbering (FDI notation)", TOOTH_BOX),
    ("spacing", "Spacing", "Missing teeth or spaces, from tooth numbering", SPACING_BOX),
    ("crowding", "Overcrowding", "Extra teeth, from tooth numbering", CROWDING_BOX),
    ("impacted", "Impacted Tooth", "Impacted tooth detection", DANGER),
]
MAX_SPACING_LINES = 3

DROP_PROMPT = "Upload a 2D panoramic X-ray (DICOM, PNG, or JPEG) to begin."

APP_VERSION = "v0.1.0"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CAMERA_POLL_MS = 33  # ~30 fps live preview
NO_CAMERA_MESSAGE = (
    "No camera found. Connect a webcam, and check that Settings › Privacy & security "
    "› Camera allows desktop apps."
)


class App(ctk.CTk):
    """Main OrthoScan AI window: sign-in screen, then the three-column workspace."""

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("OrthoScan AI")
        # Requested size is 1280x720, but the minimum height is 768,
        # so the window opens at the minimum height instead.
        self.geometry("1280x768")
        self.minsize(1024, 768)
        self.configure(fg_color=BG_MAIN)

        # Columns: sidebar 20% / canvas 55% / analysis 25%.
        # `uniform` makes the weights act as true proportions.
        self.grid_columnconfigure(0, weight=20, uniform="cols")
        self.grid_columnconfigure(1, weight=55, uniform="cols")
        self.grid_columnconfigure(2, weight=25, uniform="cols")
        self.grid_rowconfigure(0, weight=1)

        self._resize_job = None
        self._canvas_size = None
        self.placeholder_image = None
        self.canvas_image = None
        self._fitted_cache = None  # (source image, canvas size, resized copy)
        self.clinician = None

        # The X-ray currently on the canvas (PIL image, memory only - never saved here).
        self.current_image = None

        # File upload: decoding runs on a worker thread so big DICOMs don't freeze the UI.
        self._upload_executor = ThreadPoolExecutor(max_workers=1)
        self._upload_dir = DATA_DIR if DATA_DIR.is_dir() else Path.home()
        self._uploading = False

        # Report export (PDF), allowed only after the clinician approves. The PDF is
        # written on a worker thread; the clinician picks where it goes.
        self._export_executor = ThreadPoolExecutor(max_workers=1)
        documents = Path.home() / "Documents"
        self._report_dir = documents if documents.is_dir() else Path.home()
        self._exporting = False
        self._report_case = ""
        self._report_notes = ""
        self._report_message = None  # (text, is_error) shown under the approval line

        # Camera capture state. camera_mode is None, "live" or "review".
        self.camera = None
        self.camera_mode = None
        self.captured_image = None
        self._camera_job = None
        self._camera_message = None
        self._last_frame_count = None
        self._found_cameras = None
        self._probe_future = None
        self._probe_executor = ThreadPoolExecutor(max_workers=1)

        # AI analysis: models load once after sign-in, then run on each new X-ray.
        # One worker thread, so loading and analyses never overlap.
        self.analyzer = Analyzer()
        self._ai_executor = ThreadPoolExecutor(max_workers=1)
        self._ai_state = "idle"  # idle / loading / ready / unavailable
        self._ai_error = None
        self.analysis = None  # AnalysisResult for current_image, once it's done
        self.approved_at = None
        self._analysis_token = 0  # bumped on every new image/clear/sign-out
        self._analysis_error = None
        self.highlighted = set()  # finding keys to highlight on the X-ray; empty = all

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_login()

    # --- Sign-in / sign-out --------------------------------------------------
    def _show_login(self):
        # Placed over the whole window; the workspace uses grid, so they don't clash.
        self.login_view = LoginView(self, ClinicianStore(), on_success=self._on_signed_in)
        self.login_view.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _on_signed_in(self, display_name):
        self.clinician = display_name
        self.login_view.destroy()
        self._build_sidebar()
        self._build_canvas()
        self._build_analysis_panel()
        self._preload_models()

    def on_sign_out(self):
        self._close_camera()
        self.current_image = None
        self._reset_analysis()  # any analysis still running is dropped by its token
        self._uploading = False  # any upload still loading is dropped by _poll_upload
        for frame in (self.sidebar, self.canvas_frame, self.analysis_panel):
            frame.destroy()
        # Cancel after destroying: tearing down the canvas fires one last resize event.
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
            self._resize_job = None
        self.clinician = None
        self._show_login()

    # --- Left column: control sidebar ----------------------------------------
    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, fg_color=BG_SIDEBAR, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.sidebar,
            text="OrthoScan AI",
            font=ctk.CTkFont(family=FONT_FAMILY, size=26, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, padx=20, pady=(30, 4), sticky="w")

        ctk.CTkLabel(
            self.sidebar,
            text="Panoramic X-ray analysis",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            text_color=TEXT_MUTED,
        ).grid(row=1, column=0, padx=20, pady=(0, 30), sticky="w")

        actions = [
            ("Upload X-Ray", self.on_upload),
            ("Open Camera", self.on_camera),
            ("Clear Canvas", self.on_clear),
            ("Export Report", self.on_export),
        ]
        self.sidebar_buttons = {}
        for i, (text, command) in enumerate(actions, start=2):
            button = ctk.CTkButton(
                self.sidebar,
                text=text,
                command=command,
                height=44,
                corner_radius=10,
                font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
                fg_color=ACCENT,
                hover_color=ACCENT_HOVER,
            )
            button.grid(row=i, column=0, padx=20, pady=8, sticky="ew")
            self.sidebar_buttons[text] = button
        # Its label toggles to "Close Camera" while the camera is on.
        self.camera_button = self.sidebar_buttons["Open Camera"]
        # Enabled only once the findings are approved (see _show_assessment).
        self.export_button = self.sidebar_buttons["Export Report"]
        self.export_button.configure(state="disabled")

        # Spacer pushes the account block and version label to the bottom.
        bottom = len(actions) + 2
        self.sidebar.grid_rowconfigure(bottom, weight=1)

        ctk.CTkLabel(
            self.sidebar,
            text=f"Signed in as {self.clinician}",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=TEXT_PRIMARY,
            wraplength=200,
            justify="left",
            anchor="w",
        ).grid(row=bottom + 1, column=0, padx=20, pady=(0, 6), sticky="w")

        ctk.CTkButton(
            self.sidebar,
            text="Sign out",
            command=self.on_sign_out,
            height=34,
            corner_radius=10,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            fg_color="transparent",
            hover_color=BG_SURFACE,
            border_width=1,
            border_color=BORDER,
            text_color=TEXT_MUTED,
        ).grid(row=bottom + 2, column=0, padx=20, sticky="ew")

        ctk.CTkLabel(
            self.sidebar,
            text=f"{APP_VERSION} · UI preview",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=TEXT_MUTED,
        ).grid(row=bottom + 3, column=0, padx=20, pady=16, sticky="w")

    # --- Center column: diagnostic canvas ------------------------------------
    def _build_canvas(self):
        self.canvas_frame = ctk.CTkFrame(self, fg_color=BG_SURFACE, corner_radius=14)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(2, weight=1)  # the X-ray area

        header = ctk.CTkFrame(self.canvas_frame, fg_color="transparent", width=1, height=1)
        header.grid(row=0, column=0, padx=20, pady=(16, 8), sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header,
            text="Diagnostic Canvas",
            font=ctk.CTkFont(family=FONT_FAMILY, size=16, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, sticky="w")

        # Where the X-ray on the canvas came from, or what went wrong loading one.
        self.canvas_source_label = ctk.CTkLabel(
            header,
            text="",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=TEXT_MUTED,
            wraplength=380,
            justify="right",
            anchor="e",
        )
        self.canvas_source_label.grid(row=0, column=1, padx=(16, 0), sticky="e")

        # Overlay controls, shown once AI results exist: which findings to highlight
        # (cards on the right toggle them; this chip resets to all) and the on/off switch.
        bar = self.overlay_bar = ctk.CTkFrame(self.canvas_frame, fg_color="transparent", width=1, height=1)
        bar.grid(row=1, column=0, padx=20, pady=(0, 8), sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        self.all_findings_chip = ctk.CTkButton(
            bar,
            text="All findings",
            command=self._clear_highlight,
            width=96,
            height=26,
            corner_radius=13,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            border_color=BORDER,
        )
        self.all_findings_chip.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            bar,
            text="Click a finding on the right to focus on it",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=TEXT_MUTED,
            anchor="w",
            height=20,
        ).grid(row=0, column=1, padx=10, sticky="w")

        # Hides all boxes to see the bare X-ray.
        self.overlay_switch = ctk.CTkSwitch(
            bar,
            text="AI overlay",
            command=self._render_canvas,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=TEXT_MUTED,
            progress_color=ACCENT,
            switch_width=32,
            switch_height=16,
        )
        self.overlay_switch.select()
        self.overlay_switch.grid(row=0, column=2, sticky="e")
        bar.grid_remove()

        self.image_label = ctk.CTkLabel(
            self.canvas_frame,
            text=DROP_PROMPT,
            font=ctk.CTkFont(family=FONT_FAMILY, size=14),
            text_color=TEXT_MUTED,
            compound="center",
        )
        self.image_label.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew")

        # Redraw whatever the canvas shows whenever its area changes size.
        self.image_label.bind("<Configure>", self._schedule_canvas_render)

        self._build_camera_controls()

    def _build_camera_controls(self):
        """Row under the canvas, shown only while capturing from a camera."""
        controls = self.camera_controls = ctk.CTkFrame(
            self.canvas_frame, fg_color="transparent", width=1, height=1
        )
        controls.grid(row=3, column=0, padx=20, pady=(0, 16), sticky="ew")
        controls.grid_columnconfigure(1, weight=1)

        self.camera_menu = ctk.CTkOptionMenu(
            controls,
            values=[camera_name(0)],
            command=self._on_camera_selected,
            width=130,
            height=36,
            corner_radius=8,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            dropdown_font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            fg_color=BG_CARD,
            button_color=BG_CARD,
            button_hover_color=BORDER,
            dropdown_fg_color=BG_CARD,
            dropdown_hover_color=BORDER,
            text_color=TEXT_PRIMARY,
        )
        self.camera_menu.grid(row=0, column=0, sticky="w")

        self.camera_hint = ctk.CTkLabel(
            controls,
            text="Fit the X-ray inside the dashed frame.",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        self.camera_hint.grid(row=0, column=1, padx=12, sticky="w")

        def outline_button(text, command):
            return ctk.CTkButton(
                controls, text=text, command=command, width=100, height=36,
                corner_radius=10, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
                fg_color="transparent", hover_color=BG_CARD, border_width=1,
                border_color=BORDER, text_color=TEXT_PRIMARY,
            )

        def primary_button(text, command):
            return ctk.CTkButton(
                controls, text=text, command=command, width=110, height=36,
                corner_radius=10, font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                fg_color=SUCCESS, hover_color=SUCCESS_HOVER,
            )

        self.cancel_button = outline_button("Cancel", self._close_camera)
        self.capture_button = primary_button("Capture", self._capture)
        self.retake_button = outline_button("Retake", self._retake)
        self.use_photo_button = primary_button("Use photo", self._use_photo)
        for button, column in [
            (self.cancel_button, 2),
            (self.capture_button, 3),
            (self.retake_button, 2),
            (self.use_photo_button, 3),
        ]:
            button.grid(row=0, column=column, padx=(8, 0))

        self._show_camera_controls(None)

    def _show_camera_controls(self, mode):
        """None hides the row; "live" shows picker + Capture; "review" shows Retake + Use photo."""
        if mode is None:
            self.camera_controls.grid_remove()
            return
        self.camera_controls.grid()
        live = mode == "live"
        for widget in (self.camera_menu, self.camera_hint, self.cancel_button, self.capture_button):
            widget.grid() if live else widget.grid_remove()
        for widget in (self.retake_button, self.use_photo_button):
            widget.grid_remove() if live else widget.grid()

    # --- Canvas rendering ----------------------------------------------------
    def _schedule_canvas_render(self, event):
        self._canvas_size = (event.width, event.height)
        # Wrap status/error messages inside the drop zone instead of stretching it.
        width, _ = self._canvas_logical_size(event.width, event.height)
        self.image_label.configure(wraplength=max(width - 80, 200))
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(80, self._render_canvas)

    def _render_canvas(self):
        """Show the frozen capture, the current X-ray, or the empty drop zone."""
        self._resize_job = None
        if self._canvas_size is None or not self.image_label.winfo_exists():
            return
        if self.camera_mode == "live":
            # The preview loop redraws every new frame; a status/error message
            # only redraws when it changes, so refit it to the new size here.
            if self._camera_message is not None:
                self._show_canvas_message(self._camera_message)
            return
        if self.camera_mode == "review":
            self._show_on_canvas(self.captured_image)
        elif self.current_image is not None:
            show_overlay = self.analysis is not None and self.overlay_switch.get() == 1
            self._show_on_canvas(
                self.current_image,
                analysis=self.analysis if show_overlay else None,
                highlighted=self.highlighted,
            )
        else:
            self._render_placeholder(*self._canvas_size)

    def _canvas_logical_size(self, width, height):
        # CTk scales images by the window's DPI factor, so draw at logical size.
        scale = self.image_label._get_widget_scaling()
        return max(int(width / scale), 50), max(int(height / scale), 50)

    def _show_on_canvas(self, image, guide=False, analysis=None, highlighted=frozenset()):
        """Fit `image` into the canvas without stretching, on the drop-zone background.

        With `analysis`, AI boxes are drawn on the fitted copy (never on `image`);
        findings not in `highlighted` are faded (an empty set highlights all).
        """
        w, h = self._canvas_logical_size(*self._canvas_size)
        # Always fill the whole label so its size (and the layout) never shifts.
        background = Image.new("RGB", (w, h), BG_SURFACE)
        ImageDraw.Draw(background).rounded_rectangle(
            [0, 0, w - 1, h - 1], radius=12, fill=BG_PLACEHOLDER
        )
        if guide:  # live camera frame: new every time, never cached
            fitted = ImageOps.contain(image, (w, h), method=Image.Resampling.BILINEAR)
            _draw_framing_guide(fitted)
        else:
            # Reuse the resized X-ray when only the overlay changes (highlight clicks).
            cached = self._fitted_cache
            if cached and cached[0] is image and cached[1] == (w, h):
                fitted = cached[2]
            else:
                fitted = ImageOps.contain(image, (w, h), method=Image.Resampling.BILINEAR)
                self._fitted_cache = (image, (w, h), fitted)
        if analysis is not None:
            fitted = draw_analysis_overlay(
                fitted, analysis, scale=fitted.width / image.width, highlighted=highlighted
            )
        background.paste(fitted, ((w - fitted.width) // 2, (h - fitted.height) // 2))

        self.canvas_image = ctk.CTkImage(light_image=background, dark_image=background, size=(w, h))
        self.image_label.configure(image=self.canvas_image, text="")

    def _show_canvas_message(self, message):
        """Empty drop zone with a status or error message in place of the prompt."""
        if self._canvas_size is not None:
            self._render_placeholder(*self._canvas_size)
        self.image_label.configure(text=message)

    def _render_placeholder(self, width, height):
        """Draw a dashed drop-zone placeholder with PIL at the current size."""
        w, h = self._canvas_logical_size(width, height)

        img = Image.new("RGB", (w, h), BG_SURFACE)
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=12, fill=BG_PLACEHOLDER)

        inset = 10
        draw_dashed_rect(draw, (inset, inset, w - inset - 1, h - inset - 1), BORDER)

        # Simple arc hinting at a dental arch, sitting above the prompt text.
        cx, cy = w // 2, h // 2 - 40
        arch_w, arch_h = min(w // 4, 140), min(h // 8, 50)
        draw.arc(
            [cx - arch_w, cy - arch_h, cx + arch_w, cy + arch_h],
            start=200, end=340, fill=BORDER, width=3,
        )

        self.placeholder_image = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        self.image_label.configure(image=self.placeholder_image, text=DROP_PROMPT)

    # --- Right column: analysis & verification -------------------------------
    def _build_analysis_panel(self):
        panel = self.analysis_panel = ctk.CTkFrame(self, fg_color=BG_SURFACE, corner_radius=14)
        panel.grid(row=0, column=2, sticky="nsew", padx=(0, 16), pady=16)
        panel.grid_columnconfigure(0, weight=1)

        self.assessment_title = ctk.CTkLabel(
            panel,
            text="Orthodontic Assessment",
            font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold"),
            text_color=TEXT_PRIMARY,
            justify="left",
            anchor="w",
        )
        self.assessment_title.grid(row=0, column=0, padx=18, pady=(18, 2), sticky="w")

        # AI state, timing and approval. Amber while results still need review.
        self.ai_status_label = ctk.CTkLabel(
            panel,
            text="",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=TEXT_MUTED,
            wraplength=250,
            justify="left",
            anchor="w",
        )
        self.ai_status_label.grid(row=1, column=0, padx=18, pady=(0, 8), sticky="w")

        self.status_labels = {}
        self.card_details = {}
        self.card_subtitles = []
        self.card_frames = {}
        # What the cards currently show, so highlight updates only touch what changed.
        self._cards_clickable = None
        self._card_selected = {}
        for i, (key, title, subtitle, swatch) in enumerate(ASSESSMENT_CARDS, start=2):
            card = self._build_card(panel, title, subtitle, swatch)
            card.grid(row=i, column=0, padx=14, pady=4, sticky="ew")
            self.card_frames[key] = card
        self._build_fdi_chart(self.card_frames["fdi"])
        # Bind after the chart exists so every part of each card is clickable.
        for key, card in self.card_frames.items():
            _bind_click_tree(card, lambda _event, key=key: self._toggle_highlight(key))

        # Spacer pushes the verify button to the very bottom.
        bottom = len(ASSESSMENT_CARDS) + 2
        panel.grid_rowconfigure(bottom, weight=1)

        self.verify_button = ctk.CTkButton(
            panel,
            text="Verify & Approve",
            command=self.on_verify,
            height=50,
            corner_radius=10,
            font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"),
            fg_color=SUCCESS,
            hover_color=SUCCESS_HOVER,
        )
        self.verify_button.grid(row=bottom + 1, column=0, padx=14, pady=16, sticky="ew")
        # Text wraps to the panel's real width (it's 25% of the window).
        panel.bind("<Configure>", self._on_panel_resize)
        self._show_assessment()

    def _on_panel_resize(self, event):
        width = event.width / self.analysis_panel._get_widget_scaling()
        self.assessment_title.configure(wraplength=max(width - 36, 120))
        self.ai_status_label.configure(wraplength=max(width - 36, 120))
        card_text = max(width - 64, 100)  # panel and card padding on both sides
        for label in (*self.card_subtitles, *self.status_labels.values(), *self.card_details.values()):
            label.configure(wraplength=card_text)
        # Too narrow for 16 two-digit numbers: show tooth position only (8...1 | 1...8);
        # the quadrant is clear from the row and the side of the midline.
        compact = card_text < 230
        for number, cell in self.fdi_cells.items():
            cell.configure(text=number[1] if compact else number)

    def _build_card(self, parent, title, subtitle, swatch):
        card = ctk.CTkFrame(
            parent, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER,
            # CTkFrame defaults to 200x200 and re-applies it on every redraw, which made
            # cards briefly request ~450 px, squeezing the panel and clipping card text.
            width=1, height=1,
        )
        card.grid_columnconfigure(0, weight=1)

        # Title with a swatch in the colour of this finding's boxes on the X-ray.
        # Labels get small minimum heights (CTk's default is 28 px each), so four
        # cards fit at 768 px tall; wrapped text still grows them as needed.
        heading = ctk.CTkFrame(card, fg_color="transparent", width=1, height=1)
        heading.grid(row=0, column=0, padx=14, pady=(10, 0), sticky="w")
        ctk.CTkFrame(heading, width=10, height=10, corner_radius=2, fg_color=swatch).grid(
            row=0, column=0, padx=(0, 7)
        )
        ctk.CTkLabel(
            heading,
            text=title,
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
            height=20,
        ).grid(row=0, column=1, sticky="w")

        subtitle_label = ctk.CTkLabel(
            card,
            text=subtitle,
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=TEXT_MUTED,
            anchor="w",
            justify="left",
            height=16,
        )
        subtitle_label.grid(row=1, column=0, padx=14, pady=(0, 4), sticky="w")
        self.card_subtitles.append(subtitle_label)

        status = ctk.CTkLabel(
            card,
            text="",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            text_color=TEXT_MUTED,
            anchor="w",
            justify="left",
            wraplength=240,
            height=18,
        )
        status.grid(row=2, column=0, padx=14, pady=(0, 8), sticky="w")
        self.status_labels[title] = status

        # Row 3 is free for a card's own widget (the FDI chart).
        details = ctk.CTkLabel(
            card,
            text="",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=TEXT_MUTED,
            anchor="w",
            justify="left",
            wraplength=240,
            height=16,
        )
        details.grid(row=4, column=0, padx=14, pady=(0, 8), sticky="w")
        self.card_details[title] = details
        return card

    def _build_fdi_chart(self, card):
        """Dental chart: patient's right on the left, as on the X-ray. Upper jaw on top."""
        chart = ctk.CTkFrame(card, fg_color="transparent", width=1, height=1)
        chart.grid(row=3, column=0, padx=14, pady=(0, 8), sticky="ew")
        self.fdi_cells = {}
        # Small enough that 16 numbers still fit at the minimum window width.
        font = ctk.CTkFont(family=FONT_FAMILY, size=9, weight="bold")
        for row, arch in enumerate((UPPER_ARCH, LOWER_ARCH)):
            for i, number in enumerate(arch):
                column = i if i < 8 else i + 1  # column 8 is the midline
                chart.grid_columnconfigure(column, weight=1, uniform="fdi")
                cell = ctk.CTkLabel(
                    chart, text=number, font=font, text_color=BORDER, width=1, height=18
                )
                cell.grid(row=row, column=column, sticky="ew")
                self.fdi_cells[number] = cell
        # CTkFrame defaults to 200x200; size the dividers explicitly or they stretch the chart.
        ctk.CTkFrame(chart, width=1, height=1, fg_color=BORDER).grid(
            row=0, column=8, rowspan=2, sticky="ns", padx=2
        )
        ctk.CTkFrame(chart, width=1, height=1, fg_color=BORDER).grid(
            row=0, column=0, columnspan=17, sticky="sew"
        )

    def _set_card(self, title, status, color, details=""):
        self.status_labels[title].configure(text=status, text_color=color)
        label = self.card_details[title]
        label.configure(text=details)
        label.grid() if details else label.grid_remove()

    def _show_assessment(self):
        """Redraw the whole right panel from the current AI state."""
        if self.clinician is None:
            return  # signed out; the panel is gone
        result = self.analysis
        approved = self.approved_at is not None

        if self._ai_state == "unavailable":
            status, status_color = self._ai_error, DANGER
        elif self.current_image is None:
            status = "Loading AI models…" if self._ai_state == "loading" else (
                "Upload an X-ray or use the camera to start."
            )
            status_color = TEXT_MUTED
        elif self._analysis_error:
            status, status_color = self._analysis_error, DANGER
        elif result is None:
            status = "Loading AI models…" if self._ai_state == "loading" else "Analyzing X-ray…"
            status_color = PENDING
        elif approved:
            status = f"✓ Approved by {self.clinician} at {self.approved_at:%H:%M}"
            status_color = SUCCESS
            if self._exporting:
                status += " · Saving report…"
            elif self._report_message:
                message, is_error = self._report_message
                status += f" · {message}"
                status_color = DANGER if is_error else SUCCESS
        else:
            seconds = result.timings_ms["total"] / 1000
            status = f"AI suggestion — needs your review · analyzed in {seconds:.1f} s"
            status_color = PENDING
        self.ai_status_label.configure(text=status, text_color=status_color)

        if result is None:
            if self._ai_state == "unavailable" or self._analysis_error:
                card_status, color = "● Not available", DANGER
            elif self.current_image is None:
                card_status, color = "● Waiting for an X-ray", TEXT_MUTED
            else:
                card_status, color = "● Analyzing…", PENDING
            for _, title, _, _ in ASSESSMENT_CARDS:
                self._set_card(title, card_status, color)
            self._color_fdi_chart(None)
        else:
            self._show_results(result, SUCCESS if approved else PENDING)
            self._color_fdi_chart(result)

        self.verify_button.configure(
            state="normal" if result is not None and not approved else "disabled",
            text="Approved" if approved else "Verify & Approve",
        )
        # Reports only for approved findings (human-in-the-loop).
        self.export_button.configure(
            state="normal" if approved and not self._exporting else "disabled"
        )
        if result is not None:
            self.overlay_bar.grid()
        else:
            self.overlay_bar.grid_remove()
        self._refresh_highlight_ui()

    # --- Highlighting findings on the X-ray -------------------------------------
    def _toggle_highlight(self, key):
        """Card click: add/remove this finding from the highlighted set."""
        if self.analysis is None:
            return  # nothing on the X-ray to highlight yet
        self.highlighted ^= {key}
        self._refresh_highlight_ui()
        self._render_canvas()

    def _clear_highlight(self):
        """"All findings" chip: back to every box at full colour."""
        if not self.highlighted:
            return
        self.highlighted.clear()
        self._refresh_highlight_ui()
        self._render_canvas()

    def _refresh_highlight_ui(self):
        if self.clinician is None:
            return
        # Only touch widgets whose state changed: every CTk configure redraws rounded
        # shapes, and re-applying the cursor to all card widgets made each click ~120 ms.
        clickable = self.analysis is not None
        if clickable != self._cards_clickable:
            for card in self.card_frames.values():
                _set_cursor_tree(card, "hand2" if clickable else "")
            self._cards_clickable = clickable
        for key, _, _, swatch in ASSESSMENT_CARDS:
            selected = key in self.highlighted
            if self._card_selected.get(key) == selected:
                continue
            self.card_frames[key].configure(
                border_color=swatch if selected else BORDER, border_width=2 if selected else 1
            )
            self._card_selected[key] = selected
        if self.highlighted:
            self.all_findings_chip.configure(
                fg_color="transparent", border_width=1, text_color=TEXT_PRIMARY, hover_color=BG_CARD
            )
        else:
            self.all_findings_chip.configure(
                fg_color=ACCENT, border_width=0, text_color="#FFFFFF", hover_color=ACCENT_HOVER
            )

    def _show_results(self, result, color):
        spacing = result.spacing
        status = f"● {len(result.detected_numbers)} of 32 teeth detected"
        if spacing.extra_teeth:
            status += f", {spacing.extra_teeth} extra"
        # Missing numbers are listed on the Spacing card; the chart shows them dimmed.
        self._set_card("FDI Tooth Detection", status, color)

        lines = []
        if spacing.missing:
            lines.append(f"Missing tooth or space: {', '.join(spacing.missing)}")
        if spacing.third_molars_missing:
            lines.append(f"Third molar not detected: {', '.join(spacing.third_molars_missing)}")
        self._set_card(
            "Spacing",
            f"● {_plural(len(spacing.missing), 'missing tooth or space', 'missing teeth or spaces')}"
            if spacing.missing else "● No missing teeth or spaces",
            color,
            "\n".join(lines),
        )

        lines = [
            f"Extra tooth: {_plural(count, 'tooth', 'teeth')} numbered {number}"
            for number, count in spacing.extra_numbers.items()
        ]
        if len(lines) > MAX_SPACING_LINES:
            lines = lines[:MAX_SPACING_LINES] + [f"…and {len(lines) - MAX_SPACING_LINES} more"]
        self._set_card(
            "Overcrowding",
            f"● Possible overcrowding: {_plural(spacing.extra_teeth, 'extra tooth', 'extra teeth')}"
            if spacing.has_overcrowding else "● No extra teeth found",
            color,
            "\n".join(lines),
        )

        if result.impacted:
            numbers = [i.fdi or "unnumbered" for i in result.impacted]
            confidences = ", ".join(
                f"{i.fdi or 'unnumbered'} {i.detection.score:.0%}" for i in result.impacted
            )
            self._set_card(
                "Impacted Tooth",
                f"● {_plural(len(numbers), 'impacted tooth', 'impacted teeth')}: "
                f"{', '.join(numbers)}",
                color,
                f"Confidence: {confidences}",
            )
        else:
            self._set_card("Impacted Tooth", "● No impacted teeth found", color)

    def _color_fdi_chart(self, result):
        detected = result.detected_numbers if result else set()
        impacted = result.impacted_numbers if result else set()
        extra = result.spacing.extra_numbers if result else {}
        spaces = set(result.spacing.missing) if result else set()
        for number, cell in self.fdi_cells.items():
            if number in impacted:
                cell.configure(text_color=DANGER)
            elif number in extra:
                cell.configure(text_color=CROWDING_BOX)
            elif number in spaces:
                cell.configure(text_color=SPACING_BOX)
            elif number in detected:
                cell.configure(text_color=TEXT_PRIMARY)
            else:
                cell.configure(text_color=BORDER)

    # --- AI analysis ------------------------------------------------------------
    def _preload_models(self):
        """Load the models right after sign-in so the first X-ray is analyzed quickly."""
        if self._ai_state in ("loading", "ready"):
            return
        self._ai_state = "loading"
        self._ai_error = None
        self._show_assessment()
        self._poll_ai(self._ai_executor.submit(self.analyzer.load), self._on_models_loaded)

    def _poll_ai(self, future, callback, *args):
        """Check the worker from the UI thread; Tk widgets must only be touched here."""
        if not future.done():
            self.after(50, self._poll_ai, future, callback, *args)
            return
        callback(future, *args)

    def _on_models_loaded(self, future):
        try:
            future.result()
            self._ai_state = "ready"
        except AnalysisUnavailable as exc:
            self._ai_state, self._ai_error = "unavailable", str(exc)
        except Exception as exc:  # noqa: BLE001 - shown in the panel instead of crashing
            self._ai_state = "unavailable"
            self._ai_error = f"The AI models couldn't be loaded ({exc})."
        self._show_assessment()

    def _start_analysis(self):
        """Analyze current_image in the background; called whenever it changes."""
        self._reset_analysis()
        if self._ai_state == "unavailable":
            self._show_assessment()
            return
        token = self._analysis_token
        future = self._ai_executor.submit(self.analyzer.analyze, self.current_image)
        self._show_assessment()
        self._poll_ai(future, self._on_analysis_done, token)

    def _on_analysis_done(self, future, token):
        if token != self._analysis_token:
            return  # the image changed, was cleared, or the user signed out
        try:
            result = future.result()
        except AnalysisUnavailable as exc:
            self._ai_state, self._ai_error = "unavailable", str(exc)
            self._show_assessment()
            return
        except Exception as exc:  # noqa: BLE001 - one bad image shouldn't stop the app
            self._analysis_error = f"The analysis failed on this X-ray ({exc})."
            self._show_assessment()
            return
        self._ai_state = "ready"
        self.analysis = result
        self._show_assessment()
        self._render_canvas()

    def _reset_analysis(self):
        """Forget results for the previous image and ignore any analysis still running."""
        self._analysis_token += 1
        self.analysis = None
        self.approved_at = None
        self._analysis_error = None
        self.highlighted = set()  # findings to highlight on the X-ray; empty = all
        self._fitted_cache = None  # don't keep the previous X-ray in memory
        # Notes and case reference belong to this X-ray's report only.
        self._report_case = ""
        self._report_notes = ""
        self._report_message = None

    # --- Camera capture --------------------------------------------------------
    def on_camera(self):
        if self.camera_mode is None:
            self._open_camera()
        else:
            self._close_camera()

    def _open_camera(self):
        self.camera_mode = "live"
        self.camera_button.configure(text="Close Camera")
        self._show_camera_controls("live")
        # Look for other cameras in the background; the dropdown fills in when done.
        self._found_cameras = None
        self._probe_future = self._probe_executor.submit(
            list_cameras, skip=(self._selected_camera_index(),)
        )
        self._start_camera(self._selected_camera_index())

    def _start_camera(self, index):
        self._stop_camera()
        self.camera = CameraStream(index)
        self.camera.start()
        self._camera_message = None
        self._last_frame_count = None
        self.capture_button.configure(state="disabled")
        if self._camera_job is None:
            self._poll_camera()

    def _stop_camera(self, wait=False):
        if self.camera is not None:
            self.camera.stop(wait=wait)
            self.camera = None

    def _close_camera(self, wait=False):
        """Leave camera mode and release the camera. Safe to call at any time."""
        self._stop_camera(wait=wait)
        if self._camera_job is not None:
            self.after_cancel(self._camera_job)
            self._camera_job = None
        if self.camera_mode is None:
            return
        self.camera_mode = None
        self.captured_image = None
        self.camera_button.configure(text="Open Camera")
        self._show_camera_controls(None)
        self._render_canvas()

    def _selected_camera_index(self):
        return int(self.camera_menu.get().split()[-1]) - 1

    def _on_camera_selected(self, _choice):
        if self.camera_mode == "live":
            self._start_camera(self._selected_camera_index())

    def _poll_camera(self):
        """UI-thread loop: show the newest frame, or the camera's status."""
        self._camera_job = None
        if self.camera_mode != "live" or self.camera is None:
            return
        self._update_camera_choices()

        camera = self.camera
        if camera.state == CameraStream.FAILED:
            failed_only_camera = self._found_cameras == []
            self._set_camera_message(NO_CAMERA_MESSAGE if failed_only_camera else camera.error)
        elif camera.state == CameraStream.STARTING or camera.frame_count == 0:
            self._set_camera_message(f"Starting {camera.name}…")
        elif camera.frame_count != self._last_frame_count and self._canvas_size is not None:
            self._last_frame_count = camera.frame_count
            self._camera_message = None
            w, h = self._canvas_logical_size(*self._canvas_size)
            self._show_on_canvas(camera.latest_frame(max_size=(w, h)), guide=True)
            self.capture_button.configure(state="normal")

        self._camera_job = self.after(CAMERA_POLL_MS, self._poll_camera)

    def _set_camera_message(self, message):
        if message != self._camera_message:
            self._camera_message = message
            self.capture_button.configure(state="disabled")
            self._show_canvas_message(message)

    def _update_camera_choices(self):
        future = self._probe_future
        if future is None or not future.done():
            return
        self._probe_future = None
        try:
            self._found_cameras = future.result()
        except Exception:  # noqa: BLE001 - a failed probe just leaves the list as is
            self._found_cameras = []
        current = self._selected_camera_index()
        indices = sorted(set(self._found_cameras) | {current})
        self.camera_menu.configure(values=[camera_name(i) for i in indices])

    def _capture(self):
        frame = self.camera.latest_frame() if self.camera is not None else None
        if frame is None:
            return
        # Full-resolution frame, without the framing guide.
        self.captured_image = frame
        self._stop_camera()
        if self._camera_job is not None:
            self.after_cancel(self._camera_job)
            self._camera_job = None
        self.camera_mode = "review"
        self._show_camera_controls("review")
        self._render_canvas()

    def _retake(self):
        self.captured_image = None
        self.camera_mode = "live"
        self._show_camera_controls("live")
        self._start_camera(self._selected_camera_index())

    def _use_photo(self):
        # Kept in memory only; saving stays behind clinician verification.
        self.current_image = self.captured_image
        width, height = self.current_image.size
        self._set_canvas_source(f"Camera photo — {width} × {height}")
        self._close_camera()
        self._start_analysis()

    def _on_close(self):
        self._close_camera(wait=True)
        for executor in (self._probe_executor, self._upload_executor, self._ai_executor,
                         self._export_executor):
            executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    # --- File upload -----------------------------------------------------------
    def on_upload(self):
        if self._uploading:
            return
        path = filedialog.askopenfilename(
            parent=self,
            title="Upload X-ray",
            initialdir=str(self._upload_dir),
            filetypes=FILE_TYPES,
        )
        if not path:
            return  # picker cancelled
        path = Path(path)
        self._upload_dir = path.parent
        self._uploading = True
        self.sidebar_buttons["Upload X-Ray"].configure(state="disabled")
        self._set_canvas_source(f"Opening {path.name}…")
        future = self._upload_executor.submit(load_xray, path)
        self._poll_upload(future, path, self.canvas_frame)

    def _poll_upload(self, future, path, canvas_frame):
        """Check the worker from the UI thread; Tk widgets must only be touched here."""
        if not future.done():
            self.after(50, self._poll_upload, future, path, canvas_frame)
            return
        if canvas_frame is not self.canvas_frame:
            return  # signed out (and maybe back in) while loading: drop the result
        self._uploading = False
        self.sidebar_buttons["Upload X-Ray"].configure(state="normal")

        try:
            image = future.result()
        except XrayLoadError as exc:
            # Keep whatever X-ray is already on the canvas.
            self._set_canvas_source(str(exc), error=True)
            return
        except Exception:  # noqa: BLE001 - never leave the UI stuck on "Opening…"
            self._set_canvas_source(
                f"Couldn't open {path.name}: something unexpected went wrong.", error=True
            )
            raise

        self._close_camera()
        self.current_image = image
        width, height = image.size
        self._set_canvas_source(f"{path.name} — {width} × {height}")
        self._start_analysis()
        self._render_canvas()

    def _set_canvas_source(self, text, error=False):
        self.canvas_source_label.configure(
            text=text, text_color=DANGER if error else TEXT_MUTED
        )

    # --- Button handlers (stubs until the backend exists) --------------------
    def on_clear(self):
        self._close_camera()
        self.current_image = None
        self._reset_analysis()
        self._set_canvas_source("")
        self._show_assessment()
        self._render_canvas()

    # --- Report export -----------------------------------------------------------
    def on_export(self):
        """Ask for the case reference and notes, then where to save the PDF."""
        if self.analysis is None or self.approved_at is None or self._exporting:
            return  # only approved findings can be exported
        ExportDialog(
            self, on_save=self._save_report,
            case_reference=self._report_case, notes=self._report_notes,
        )

    def _save_report(self, case_reference, notes):
        # Keep what was typed so a second export starts from it.
        self._report_case, self._report_notes = case_reference, notes
        stem = _safe_filename(case_reference) or f"{self.approved_at:%Y-%m-%d_%H%M}"
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save report",
            initialdir=str(self._report_dir),
            initialfile=f"OrthoScan_report_{stem}.pdf",
            defaultextension=".pdf",
            filetypes=[("PDF document", "*.pdf")],
        )
        if not path:
            return  # save dialog cancelled
        path = Path(path)
        self._report_dir = path.parent
        data = ReportData(
            image=self.current_image,
            analysis=self.analysis,
            clinician=self.clinician,
            approved_at=self.approved_at,
            case_reference=case_reference,
            notes=notes,
            app_version=APP_VERSION,
        )
        self._exporting = True
        self._report_message = None
        self._show_assessment()
        future = self._export_executor.submit(save_report_pdf, data, path)
        self._poll_export(future, path, self._analysis_token)

    def _poll_export(self, future, path, token):
        if not future.done():
            self.after(50, self._poll_export, future, path, token)
            return
        if token != self._analysis_token:
            return  # new X-ray, Clear, or sign-out while saving: the old message is moot
        self._exporting = False
        try:
            future.result()
            self._report_message = (f"Report saved: {path.name}", False)
        except Exception as exc:  # noqa: BLE001 - e.g. file open in a PDF viewer, no permission
            self._report_message = (f"Couldn't save the report ({exc})", True)
        self._show_assessment()

    def on_verify(self):
        """Clinician sign-off on the AI suggestions (in memory; nothing is saved yet)."""
        if self.analysis is None or self.approved_at is not None:
            return
        self.approved_at = datetime.now()
        self._show_assessment()


def _safe_filename(text, limit=60):
    """Case reference -> safe file-name part (letters, digits, dash, underscore)."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in text.strip())
    return "_".join(part for part in cleaned.split("_") if part)[:limit]


def _bind_click_tree(widget, handler):
    """Bind a click on a widget and every widget inside it.

    Uses plain tkinter on each underlying widget: CTk's own bind() forwards to its
    inner widgets, so combining both would fire the handler twice. Children come
    from tkinter.Misc.winfo_children because CTkFrame's override hides its
    background canvas, which is what a click on a card's empty area actually hits.
    """
    tkinter.Misc.bind(widget, "<Button-1>", handler, "+")
    for child in tkinter.Misc.winfo_children(widget):
        _bind_click_tree(child, handler)


def _set_cursor_tree(widget, cursor):
    try:
        tkinter.Misc.configure(widget, cursor=cursor)
    except tkinter.TclError:
        pass  # a widget type without a cursor option
    for child in tkinter.Misc.winfo_children(widget):  # includes CTkFrame canvases
        _set_cursor_tree(child, cursor)


def _plural(count, singular, plural=None):
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def _draw_framing_guide(image):
    """Dashed 2:1 box (panoramic X-ray shape) drawn onto a preview frame in place."""
    w, h = image.size
    guide_w = min(int(w * 0.88), int(h * 0.88) * 2)
    guide_h = guide_w // 2
    left, top = (w - guide_w) // 2, (h - guide_h) // 2
    draw_dashed_rect(
        ImageDraw.Draw(image), (left, top, left + guide_w, top + guide_h), TEXT_PRIMARY
    )


if __name__ == "__main__":
    app = App()
    app.mainloop()
