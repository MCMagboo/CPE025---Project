"""Sign-in screen shown before the main OrthoScan AI workspace."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import customtkinter as ctk

from auth import AccountError, ClinicianStore
from theme import (
    ACCENT,
    ACCENT_HOVER,
    BG_MAIN,
    BG_SIDEBAR,
    BG_SURFACE,
    BORDER,
    DANGER,
    FONT_FAMILY,
    TEXT_MUTED,
    TEXT_PRIMARY,
)

FORM_WIDTH = 360

SIGN_IN_FIELDS = [
    ("username", "Username", False),
    ("password", "Password", True),
]
SETUP_FIELDS = [
    ("name", "Full name", False),
    ("username", "Username", False),
    ("password", "Password", True),
    ("confirm", "Confirm password", True),
]


class LoginView(ctk.CTkFrame):
    """Full-window sign-in / sign-up screen.

    A text link under the form switches between signing in and creating an
    account. If no clinician accounts exist on this computer yet, it opens on
    sign-up with no link. Password hashing runs on a worker thread so the
    window stays responsive while it checks.
    """

    def __init__(self, master, store: ClinicianStore, on_success):
        super().__init__(master, fg_color=BG_MAIN, corner_radius=0)
        self.store = store
        self.on_success = on_success
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._busy = False
        self._has_accounts = store.has_accounts()
        self._signup_mode = not self._has_accounts
        self.switch_link = None

        # Brand panel 40% / form 60%.
        self.grid_columnconfigure(0, weight=2, uniform="login")
        self.grid_columnconfigure(1, weight=3, uniform="login")
        self.grid_rowconfigure(0, weight=1)

        self._build_brand_panel()
        self._build_form()

    # --- Left: brand panel (same color as the workspace sidebar) -------------
    def _build_brand_panel(self):
        panel = ctk.CTkFrame(self, fg_color=BG_SIDEBAR, corner_radius=0)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        # Empty rows above and below center the text vertically.
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            panel,
            text="OrthoScan AI",
            font=ctk.CTkFont(family=FONT_FAMILY, size=34, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).grid(row=1, column=0, padx=48, sticky="w")

        ctk.CTkLabel(
            panel,
            text=(
                "Decision support for panoramic X-rays. "
                "Every finding waits for your review before it is saved."
            ),
            font=ctk.CTkFont(family=FONT_FAMILY, size=15),
            text_color=TEXT_MUTED,
            wraplength=320,
            justify="left",
            anchor="w",
        ).grid(row=2, column=0, padx=48, pady=(12, 0), sticky="w")

    # --- Right: credentials form ---------------------------------------------
    def _build_form(self):
        # No sticky, so the form keeps its width and sits centered in the column.
        form = self.form = ctk.CTkFrame(self, fg_color="transparent")
        form.grid(row=0, column=1)

        if self._signup_mode and not self._has_accounts:
            heading = "Create the first clinician account"
            subtitle = "No clinician accounts exist on this computer yet. Create one to continue."
            fields = SETUP_FIELDS
            self._button_text = ("Create account", "Creating account…")
        elif self._signup_mode:
            heading = "Create an account"
            subtitle = "Add a clinician account for this computer."
            fields = SETUP_FIELDS
            self._button_text = ("Create account", "Creating account…")
        else:
            heading = "Sign in"
            subtitle = "Use your clinician account to open the workspace."
            fields = SIGN_IN_FIELDS
            self._button_text = ("Sign in", "Signing in…")

        ctk.CTkLabel(
            form,
            text=heading,
            font=ctk.CTkFont(family=FONT_FAMILY, size=24, weight="bold"),
            text_color=TEXT_PRIMARY,
            wraplength=FORM_WIDTH,
            justify="left",
            anchor="w",
        ).pack(fill="x")

        ctk.CTkLabel(
            form,
            text=subtitle,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            text_color=TEXT_MUTED,
            wraplength=FORM_WIDTH,
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(4, 24))

        self.entries = {}
        for key, label, secret in fields:
            ctk.CTkLabel(
                form,
                text=label,
                font=ctk.CTkFont(family=FONT_FAMILY, size=12),
                text_color=TEXT_MUTED,
                anchor="w",
            ).pack(fill="x")
            entry = ctk.CTkEntry(
                form,
                width=FORM_WIDTH,
                height=40,
                corner_radius=8,
                fg_color=BG_SURFACE,
                border_color=BORDER,
                text_color=TEXT_PRIMARY,
                font=ctk.CTkFont(family=FONT_FAMILY, size=14),
                show="•" if secret else "",
            )
            entry.pack(pady=(2, 14))
            entry.bind("<Return>", self._submit)
            self.entries[key] = entry

        self.error_label = ctk.CTkLabel(
            form,
            text="",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=DANGER,
            wraplength=FORM_WIDTH,
            justify="left",
            anchor="w",
        )
        self.error_label.pack(fill="x")

        self.submit_button = ctk.CTkButton(
            form,
            text=self._button_text[0],
            command=self._submit,
            width=FORM_WIDTH,
            height=44,
            corner_radius=10,
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
        )
        self.submit_button.pack(pady=(8, 0))

        # Nothing to sign in to before the first account exists, so no link then.
        self.switch_link = None
        if self._has_accounts:
            self._build_switch_link(form)

        first_entry = next(iter(self.entries.values()))
        # The form may be rebuilt (link clicked quickly) before this fires.
        self.after(100, lambda: first_entry.winfo_exists() and first_entry.focus_set())

    def _build_switch_link(self, form):
        if self._signup_mode:
            question, link_text = "Already have an account?", "Sign in"
        else:
            question, link_text = "Don't have an account?", "Create one"

        row = ctk.CTkFrame(form, fg_color="transparent")
        row.pack(pady=(16, 0))

        ctk.CTkLabel(
            row,
            text=question,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            text_color=TEXT_MUTED,
        ).pack(side="left")

        plain = ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold")
        underlined = ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold", underline=True)
        self.switch_link = ctk.CTkLabel(
            row, text=link_text, font=plain, text_color=ACCENT, cursor="hand2"
        )
        self.switch_link.pack(side="left", padx=(6, 0))
        self.switch_link.bind("<Button-1>", self._switch_mode)
        self.switch_link.bind(
            "<Enter>", lambda _e: self._busy or self.switch_link.configure(font=underlined)
        )
        self.switch_link.bind("<Leave>", lambda _e: self.switch_link.configure(font=plain))

    def _switch_mode(self, _event=None):
        if self._busy:
            return
        self._signup_mode = not self._signup_mode
        self.form.destroy()
        self._build_form()

    # --- Submitting ----------------------------------------------------------
    def _submit(self, _event=None):
        if self._busy:
            return
        values = {key: entry.get() for key, entry in self.entries.items()}

        if self._signup_mode:
            if values["password"] != values["confirm"]:
                self._show_error("The passwords don't match.")
                return
            task = partial(
                self.store.create, values["username"], values["name"], values["password"]
            )
        else:
            if not values["username"].strip() or not values["password"]:
                self._show_error("Enter your username and password.")
                return
            task = partial(self.store.verify, values["username"], values["password"])

        self._show_error("")
        self._set_busy(True)
        self._poll(self._executor.submit(task))

    def _poll(self, future):
        """Check the worker from the UI thread; Tk widgets must only be touched here."""
        if not future.done():
            self.after(50, self._poll, future)
            return
        self._set_busy(False)

        try:
            display_name = future.result()
        except AccountError as exc:
            self._show_error(str(exc))
            return
        except (OSError, ValueError) as exc:
            self._show_error(f"The account file at {self.store.path} couldn't be used: {exc}")
            return

        if display_name is None:
            self.entries["password"].delete(0, "end")
            self.entries["password"].focus_set()
            self._show_error("The username or password is incorrect.")
            return

        self._executor.shutdown(wait=False)
        self.on_success(display_name)

    def _set_busy(self, busy):
        self._busy = busy
        state = "disabled" if busy else "normal"
        for entry in self.entries.values():
            entry.configure(state=state)
        self.submit_button.configure(state=state, text=self._button_text[1 if busy else 0])
        if self.switch_link is not None:
            self.switch_link.configure(text_color=TEXT_MUTED if busy else ACCENT)

    def _show_error(self, message):
        self.error_label.configure(text=message)
