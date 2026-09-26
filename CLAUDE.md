# 🦷 OrthoScan AI — Project Context & Engineering Guide

> **Read this file first.** It is the source of truth for architecture, constraints, and coding standards in this workspace.

---

## 1. Project Overview

| Field | Details |
|---|---|
| **Name** | OrthoScan AI |
| **Purpose** | A vision-based deep learning desktop application for automated orthodontic assessment using panoramic dental X-rays. |
| **Role** | Strictly a **clinician-verified decision-support tool** — never an autonomous diagnostic system. |

### Core Capabilities

1. **FDI Tooth Detection & Numbering** — Locate each tooth and assign its FDI (ISO 3950) number.
2. **Interdental Spacing & Overcrowding Analysis** — Measure gaps and crowding between adjacent teeth.
3. **Displacement Severity Classification** — 5-tier scale:

   | Tier | Label |
   |:---:|---|
   | 0 | Perfect |
   | 1 | Minimal |
   | 2 | Moderate |
   | 3 | Severe |
   | 4 | Very Severe |

---

## 2. Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| **Frontend UI** | Python + `customtkinter` | Modern, dark-mode, flat-design UI |
| **Image Handling (UI)** | `Pillow` | Loading/rendering images into the canvas |
| **Preprocessing** | `OpenCV` | CLAHE contrast enhancement, min-max normalization |
| **AI / Inference** | `PyTorch` or `TensorFlow` | YOLOv8, RT-DETR, or Mask R-CNN models |
| **Environment** | Local desktop execution | Docker containerization planned for later |

---

## 3. Project Directory Structure

```text
orthoscan-ai/
├── assets/            # UI assets, icons, and static images
├── models/            # Saved weights and architecture files for DL models
├── data/              # Local storage for anonymized X-rays (DICOM, PNG, JPEG)
├── src/               # All Python source code
│   └── main.py        #   UI scripts, inference scripts, preprocessing modules
├── requirements.txt   # Python dependencies (keep up to date!)
└── CLAUDE.md          # This file
```

> ⚠️ `/data` must contain **anonymized** images only. Never commit patient-identifiable information.

---

## 4. Strict Engineering Constraints

| Constraint | Requirement |
|---|---|
| ⏱️ **Performance** | Inference response time must be between **1,000 ms and 2,000 ms**. |
| 💾 **Resource Limits** | Peak RAM usage must remain **below 4,000 MB (4 GB)**. |
| 🩺 **Safety (HITL)** | Human-in-the-Loop is mandatory: **all AI outputs must be routed to a UI panel for final clinician verification before saving.** |

- No result may be persisted, exported, or treated as final without explicit clinician confirmation in the UI.
- Consider memory footprint when choosing model sizes, batch sizes, and image resolutions; release large tensors/arrays when no longer needed.

---

## 5. Coding Guidelines & UI Standards

### 🏗️ Architecture
- Use **Object-Oriented Programming (OOP)**.
- The UI **must be decoupled** from heavy AI processing. Run inference off the main thread (e.g., `threading` / `concurrent.futures`) and marshal results back to the UI safely (e.g., via `widget.after(...)`) so the GUI never freezes.

### 🎨 UI Aesthetic
- Strictly maintain a **modern, dark-theme** layout using `customtkinter`.
- **Avoid** standard, outdated Tkinter widgets — prefer `CTk*` equivalents.
- Use the standard **3-column layout**:

  | Sidebar | Image Canvas | Results Panel |
  |---|---|---|
  | Navigation, file loading, actions | X-ray display with overlays | AI findings + clinician verification controls |

### 📦 Dependencies
- **Always update `requirements.txt`** when adding new packages.

### 🧩 Modularity
Separate responsibilities into distinct functions or classes:

1. **Image Ingestion** — loading DICOM / PNG / JPEG
2. **OpenCV Preprocessing** — CLAHE, min-max normalization
3. **AI Inference** — model loading and prediction
4. **UI Updates** — rendering results and verification controls

---

## ✅ Directive for Claude

> **Claude must review this `CLAUDE.md` file before generating new code or proposing architectural changes for this project.** All contributions must comply with the constraints, architecture, and standards defined above. If a requested change conflicts with this document, flag the conflict before proceeding.
