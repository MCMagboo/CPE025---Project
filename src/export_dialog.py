"""Export dialog: optional case reference + clinician notes before saving the PDF report."""

import customtkinter as ctk

from report import MAX_NOTES_CHARS
from theme import (
    BG_CARD,
    BG_MAIN,
    BG_SURFACE,
    BORDER,
    DANGER,
    FONT_FAMILY,
    SUCCESS,
    SUCCESS_HOVER,
    TEXT_MUTED,
    TEXT_PRIMARY,
)


class ExportDialog(ctk.CTkToplevel):
    """Modal dialog; calls on_save(case_reference, notes) when the clinician confirms."""

    def __init__(self, master, on_save, case_reference="", notes=""):
        super().__init__(master, fg_color=BG_MAIN)
        self.on_save = on_save
        self.title("Export report")
        self.resizable(False, False)
        self.transient(master)
        self.grid_columnconfigure(0, weight=1)

        def label(text, row, muted=False, size=13, pady=(0, 4)):
            widget = ctk.CTkLabel(
                self, text=text, anchor="w", justify="left", wraplength=460,
                font=ctk.CTkFont(family=FONT_FAMILY, size=size, weight="normal" if muted else "bold"),
                text_color=TEXT_MUTED if muted else TEXT_PRIMARY,
            )
            widget.grid(row=row, column=0, padx=24, pady=pady, sticky="w")
            return widget

        label("Export report", 0, size=18, pady=(20, 2))
        label("A one-page PDF of the approved findings, with your notes.", 1, muted=True, pady=(0, 16))

        label("Case reference (optional)", 2)
        self.case_entry = ctk.CTkEntry(
            self, width=460, height=36, corner_radius=8, fg_color=BG_SURFACE,
            border_color=BORDER, text_color=TEXT_PRIMARY,
            placeholder_text="e.g. Case 2026-114",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
        )
        self.case_entry.grid(row=3, column=0, padx=24, sticky="ew")
        if case_reference:
            self.case_entry.insert(0, case_reference)
        label("Use a case number, not the patient's name.", 4, muted=True, size=11, pady=(2, 14))

        label("Clinician notes", 5)
        self.notes_box = ctk.CTkTextbox(
            self, width=460, height=170, corner_radius=8, fg_color=BG_SURFACE,
            border_color=BORDER, border_width=1, text_color=TEXT_PRIMARY, wrap="word",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
        )
        self.notes_box.grid(row=6, column=0, padx=24, sticky="ew")
        if notes:
            self.notes_box.insert("1.0", notes)
        self.notes_box.bind("<KeyRelease>", self._update_counter)
        self.counter = label("", 7, muted=True, size=11, pady=(2, 16))
        self._update_counter()

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.grid(row=8, column=0, padx=24, pady=(0, 20), sticky="e")
        ctk.CTkButton(
            buttons, text="Cancel", command=self.destroy, width=100, height=36, corner_radius=10,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13), fg_color="transparent",
            hover_color=BG_CARD, border_width=1, border_color=BORDER, text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, padx=(0, 8))
        self.save_button = ctk.CTkButton(
            buttons, text="Save PDF…", command=self._save, width=120, height=36, corner_radius=10,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            fg_color=SUCCESS, hover_color=SUCCESS_HOVER,
        )
        self.save_button.grid(row=0, column=1)

        self.bind("<Escape>", lambda _e: self.destroy())
        self._center_over(master)
        self.after(50, self._make_modal)

    def _make_modal(self):
        # grab_set fails if the window isn't viewable yet, so it runs just after showing.
        try:
            self.grab_set()
        except Exception:  # noqa: BLE001 - modality is a convenience, not required
            pass
        self.case_entry.focus_set()

    def _center_over(self, master):
        self.update_idletasks()
        x = master.winfo_rootx() + (master.winfo_width() - self.winfo_reqwidth()) // 2
        y = master.winfo_rooty() + (master.winfo_height() - self.winfo_reqheight()) // 3
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def notes(self):
        return self.notes_box.get("1.0", "end").strip()

    def _update_counter(self, _event=None):
        text = self.notes()
        if len(text) > MAX_NOTES_CHARS:  # keep within what fits on the page
            self.notes_box.delete("1.0", "end")
            self.notes_box.insert("1.0", text[:MAX_NOTES_CHARS])
            text = text[:MAX_NOTES_CHARS]
        self.counter.configure(
            text=f"{len(text)} / {MAX_NOTES_CHARS} characters",
            text_color=DANGER if len(text) >= MAX_NOTES_CHARS else TEXT_MUTED,
        )

    def _save(self):
        case_reference, notes = self.case_entry.get().strip(), self.notes()
        self.destroy()
        self.on_save(case_reference, notes)
