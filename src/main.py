"""OrthoScan AI - clinical desktop UI shell.

A clinician signs in first (see login.py), then the three-column workspace
opens. The AI backend (tooth detection, spacing analysis, displacement
grading) is not implemented yet; workspace button handlers are stubs.
"""

import customtkinter as ctk
from PIL import Image, ImageDraw

from auth import ClinicianStore
from login import LoginView
from theme import (
    ACCENT,
    ACCENT_HOVER,
    BG_CARD,
    BG_MAIN,
    BG_PLACEHOLDER,
    BG_SIDEBAR,
    BG_SURFACE,
    BORDER,
    FONT_FAMILY,
    PENDING,
    SUCCESS,
    SUCCESS_HOVER,
    TEXT_MUTED,
    TEXT_PRIMARY,
)

SEVERITY_TIERS = ["Perfect", "Mild", "Moderate", "Severe", "Very Severe"]


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
        self.placeholder_image = None
        self.clinician = None

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

    def on_sign_out(self):
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
            self._resize_job = None
        for frame in (self.sidebar, self.canvas_frame, self.analysis_panel):
            frame.destroy()
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
            ("Clear Canvas", self.on_clear),
            ("Export Report", self.on_export),
        ]
        for i, (text, command) in enumerate(actions, start=2):
            ctk.CTkButton(
                self.sidebar,
                text=text,
                command=command,
                height=44,
                corner_radius=10,
                font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
                fg_color=ACCENT,
                hover_color=ACCENT_HOVER,
            ).grid(row=i, column=0, padx=20, pady=8, sticky="ew")

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
            text="v0.1.0 · UI preview",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=TEXT_MUTED,
        ).grid(row=bottom + 3, column=0, padx=20, pady=16, sticky="w")

    # --- Center column: diagnostic canvas ------------------------------------
    def _build_canvas(self):
        self.canvas_frame = ctk.CTkFrame(self, fg_color=BG_SURFACE, corner_radius=14)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self.canvas_frame,
            text="Diagnostic Canvas",
            font=ctk.CTkFont(family=FONT_FAMILY, size=16, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, padx=20, pady=(16, 8), sticky="w")

        self.image_label = ctk.CTkLabel(
            self.canvas_frame,
            text="Upload a 2D panoramic X-ray (DICOM, PNG, or JPEG) to begin.",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14),
            text_color=TEXT_MUTED,
            compound="center",
        )
        self.image_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

        # Regenerate the placeholder whenever the canvas area changes size.
        self.image_label.bind("<Configure>", self._schedule_placeholder_resize)

    def _schedule_placeholder_resize(self, event):
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(80, self._render_placeholder, event.width, event.height)

    def _render_placeholder(self, width, height):
        """Draw a dashed drop-zone placeholder with PIL at the current size."""
        self._resize_job = None
        # CTk scales images by the window's DPI factor, so draw at logical size.
        scale = self.image_label._get_widget_scaling()
        w = max(int(width / scale), 50)
        h = max(int(height / scale), 50)

        img = Image.new("RGB", (w, h), BG_SURFACE)
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=12, fill=BG_PLACEHOLDER)

        # Dashed border.
        inset, dash, gap = 10, 14, 10
        left, top, right, bottom = inset, inset, w - inset - 1, h - inset - 1
        for x in range(left, right, dash + gap):
            x_end = min(x + dash, right)
            draw.line([(x, top), (x_end, top)], fill=BORDER, width=2)
            draw.line([(x, bottom), (x_end, bottom)], fill=BORDER, width=2)
        for y in range(top, bottom, dash + gap):
            y_end = min(y + dash, bottom)
            draw.line([(left, y), (left, y_end)], fill=BORDER, width=2)
            draw.line([(right, y), (right, y_end)], fill=BORDER, width=2)

        # Simple arc hinting at a dental arch, sitting above the prompt text.
        cx, cy = w // 2, h // 2 - 40
        arch_w, arch_h = min(w // 4, 140), min(h // 8, 50)
        draw.arc(
            [cx - arch_w, cy - arch_h, cx + arch_w, cy + arch_h],
            start=200, end=340, fill=BORDER, width=3,
        )

        self.placeholder_image = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        self.image_label.configure(image=self.placeholder_image)

    # --- Right column: analysis & verification -------------------------------
    def _build_analysis_panel(self):
        self.analysis_panel = ctk.CTkFrame(self, fg_color=BG_SURFACE, corner_radius=14)
        self.analysis_panel.grid(row=0, column=2, sticky="nsew", padx=(0, 16), pady=16)
        self.analysis_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.analysis_panel,
            text="Orthodontic Assessment",
            font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, padx=18, pady=(18, 12), sticky="w")

        self.status_labels = {}
        cards = [
            ("FDI Tooth Detection", "Tooth numbering (FDI notation)"),
            ("Spacing & Overcrowding", "Inter-dental spacing analysis"),
            ("Displacement Severity", "5-tier displacement grading"),
        ]
        for i, (title, subtitle) in enumerate(cards, start=1):
            card = self._build_card(self.analysis_panel, title, subtitle)
            card.grid(row=i, column=0, padx=14, pady=6, sticky="ew")

        # Placeholder severity scale on the last card.
        self._build_severity_scale(card)

        # Spacer pushes the verify button to the very bottom.
        self.analysis_panel.grid_rowconfigure(len(cards) + 1, weight=1)

        self.verify_button = ctk.CTkButton(
            self.analysis_panel,
            text="Verify & Approve",
            command=self.on_verify,
            height=50,
            corner_radius=10,
            font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"),
            fg_color=SUCCESS,
            hover_color=SUCCESS_HOVER,
        )
        self.verify_button.grid(row=len(cards) + 2, column=0, padx=14, pady=16, sticky="ew")

    def _build_card(self, parent, title, subtitle):
        card = ctk.CTkFrame(
            parent, fg_color=BG_CARD, corner_radius=10, border_width=1, border_color=BORDER
        )
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        ).grid(row=0, column=0, padx=14, pady=(12, 0), sticky="w")

        ctk.CTkLabel(
            card,
            text=subtitle,
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=TEXT_MUTED,
            anchor="w",
        ).grid(row=1, column=0, padx=14, pady=(0, 4), sticky="w")

        status = ctk.CTkLabel(
            card,
            text="● Pending",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            text_color=PENDING,
            anchor="w",
        )
        status.grid(row=2, column=0, padx=14, pady=(0, 12), sticky="w")
        self.status_labels[title] = status
        return card

    def _build_severity_scale(self, card):
        """Dimmed 5-tier scale; will highlight the detected tier once the model runs."""
        scale = ctk.CTkFrame(card, fg_color="transparent")
        scale.grid(row=3, column=0, padx=14, pady=(0, 14), sticky="ew")
        self.severity_segments = []
        for i, tier in enumerate(SEVERITY_TIERS):
            scale.grid_columnconfigure(i, weight=1, uniform="tiers")
            segment = ctk.CTkFrame(scale, height=6, corner_radius=3, fg_color=BORDER)
            segment.grid(row=0, column=i, padx=2, sticky="ew")
            self.severity_segments.append(segment)
        ctk.CTkLabel(
            scale, text=SEVERITY_TIERS[0], font=ctk.CTkFont(family=FONT_FAMILY, size=10),
            text_color=TEXT_MUTED,
        ).grid(row=1, column=0, columnspan=2, sticky="w")
        ctk.CTkLabel(
            scale, text=SEVERITY_TIERS[-1], font=ctk.CTkFont(family=FONT_FAMILY, size=10),
            text_color=TEXT_MUTED,
        ).grid(row=1, column=3, columnspan=2, sticky="e")

    # --- Button handlers (stubs until the backend exists) --------------------
    def on_upload(self):
        pass

    def on_clear(self):
        pass

    def on_export(self):
        pass

    def on_verify(self):
        pass


if __name__ == "__main__":
    app = App()
    app.mainloop()
