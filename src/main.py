"""OrthoScan AI - clinical desktop UI shell.

A clinician signs in first (see login.py), then the three-column workspace
opens. A printed X-ray can be captured with a webcam (see camera.py). The AI
backend (tooth detection, spacing analysis, displacement grading) is not
implemented yet; the remaining workspace button handlers are stubs.
"""

from concurrent.futures import ThreadPoolExecutor

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageOps

from auth import ClinicianStore
from camera import CameraStream, camera_name, list_cameras
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

DROP_PROMPT = "Upload a 2D panoramic X-ray (DICOM, PNG, or JPEG) to begin."

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
        self.clinician = None

        # The X-ray currently on the canvas (PIL image, memory only - never saved here).
        self.current_image = None

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

    def on_sign_out(self):
        self._close_camera()
        self.current_image = None
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
            text=DROP_PROMPT,
            font=ctk.CTkFont(family=FONT_FAMILY, size=14),
            text_color=TEXT_MUTED,
            compound="center",
        )
        self.image_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

        # Redraw whatever the canvas shows whenever its area changes size.
        self.image_label.bind("<Configure>", self._schedule_canvas_render)

        self._build_camera_controls()

    def _build_camera_controls(self):
        """Row under the canvas, shown only while capturing from a camera."""
        controls = self.camera_controls = ctk.CTkFrame(self.canvas_frame, fg_color="transparent")
        controls.grid(row=2, column=0, padx=20, pady=(0, 16), sticky="ew")
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
        if self._canvas_size is None:
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
            self._show_on_canvas(self.current_image)
        else:
            self._render_placeholder(*self._canvas_size)

    def _canvas_logical_size(self, width, height):
        # CTk scales images by the window's DPI factor, so draw at logical size.
        scale = self.image_label._get_widget_scaling()
        return max(int(width / scale), 50), max(int(height / scale), 50)

    def _show_on_canvas(self, image, guide=False):
        """Fit `image` into the canvas without stretching, on the drop-zone background."""
        w, h = self._canvas_logical_size(*self._canvas_size)
        # Always fill the whole label so its size (and the layout) never shifts.
        background = Image.new("RGB", (w, h), BG_SURFACE)
        ImageDraw.Draw(background).rounded_rectangle(
            [0, 0, w - 1, h - 1], radius=12, fill=BG_PLACEHOLDER
        )
        fitted = ImageOps.contain(image, (w, h), method=Image.Resampling.BILINEAR)
        if guide:
            _draw_framing_guide(fitted)
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
        _draw_dashed_rect(draw, (inset, inset, w - inset - 1, h - inset - 1), BORDER)

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
        self._close_camera()

    def _on_close(self):
        self._close_camera(wait=True)
        self._probe_executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

    # --- Button handlers (stubs until the backend exists) --------------------
    def on_upload(self):
        pass

    def on_clear(self):
        self._close_camera()
        self.current_image = None
        self._render_canvas()

    def on_export(self):
        pass

    def on_verify(self):
        pass


def _draw_dashed_rect(draw, box, color, width=2, dash=14, gap=10):
    left, top, right, bottom = box
    for x in range(left, right, dash + gap):
        x_end = min(x + dash, right)
        draw.line([(x, top), (x_end, top)], fill=color, width=width)
        draw.line([(x, bottom), (x_end, bottom)], fill=color, width=width)
    for y in range(top, bottom, dash + gap):
        y_end = min(y + dash, bottom)
        draw.line([(left, y), (left, y_end)], fill=color, width=width)
        draw.line([(right, y), (right, y_end)], fill=color, width=width)


def _draw_framing_guide(image):
    """Dashed 2:1 box (panoramic X-ray shape) drawn onto a preview frame in place."""
    w, h = image.size
    guide_w = min(int(w * 0.88), int(h * 0.88) * 2)
    guide_h = guide_w // 2
    left, top = (w - guide_w) // 2, (h - guide_h) // 2
    _draw_dashed_rect(
        ImageDraw.Draw(image), (left, top, left + guide_w, top + guide_h), TEXT_PRIMARY
    )


if __name__ == "__main__":
    app = App()
    app.mainloop()
