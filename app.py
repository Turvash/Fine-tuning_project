"""
TkSummarizer — Offline Text Summarizer GUI (Tkinter)
----------------------------------------------------

Features
- Choose among BART, Pegasus, and T5 models for summarization.
- Works fully offline after the first successful download.
- Caches each model in ./models/<model_name>/ so you never re-download.
- Threaded inference so the UI stays responsive.
- Adjustable parameters (max length, min length, beam size).
- Load text from file and save summaries.

First run (one-time downloads)
- Make sure you have internet ON *only for the first time per model* so it can download.
- After that, everything loads from the local ./models folder.

Folder layout (created automatically)
./models/
  bart-large-cnn/
  pegasus-xsum/
  t5-base/

Python & package requirements
- Python 3.9+
- pip install: transformers torch sentencepiece
  (OPTIONAL but recommended) pip install accelerate

Run
- python app.py

Notes
- T5 expects the prefix "summarize: " before the input.
- Pegasus requires the 'sentencepiece' package.
- If you are completely offline and the model folder doesn't exist yet, the app will show an error asking you to connect once to download.

"""

import os
import sys
import threading
import queue
from pathlib import Path
from dataclasses import dataclass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Suppress HF/transformers info logs in console
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
except Exception as e:
    print("\n[ERROR] Missing dependencies. Please run:\n  pip install transformers torch sentencepiece\n")
    raise


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "TkSummarizer — Offline (BART / Pegasus / T5)"
BASE_MODELS_DIR = Path("models")  # All model folders stored here
BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_MODELS = {
    "BART (facebook/bart-large-cnn)": {
        "hub_id": "facebook/bart-large-cnn",
        "local_dir": BASE_MODELS_DIR / "bart-large-cnn",
        "needs_prefix": False,
    },
    "Pegasus (google/pegasus-xsum)": {
        "hub_id": "CyrexPro/pegasus-xsum-finetuned-cnn_dailymail",
        "local_dir": BASE_MODELS_DIR / "pegasus-cnn_dailymail",
        "needs_prefix": False,
    },
    "T5 (t5-base)": {
        "hub_id": "Turvash/t5-cnn-dm-summarizer",
        "local_dir": BASE_MODELS_DIR / "t5-base",
        "needs_prefix": True,  # T5 likes "summarize: "
    },
}


@dataclass
class SummarizeParams:
    max_length: int = 180
    min_length: int = 30
    num_beams: int = 4
    do_sample: bool = False


class ModelManager:
    """Handle on-demand download, local caching and loading of models/tokenizers."""

    def __init__(self, device_preference: str = "auto"):
        # Resolve device index for HF pipeline
        if device_preference == "cpu":
            self.device = -1
        else:
            self.device = 0 if torch.cuda.is_available() else -1

        self.loaded = {}  # key -> (pipeline, needs_prefix)

    def _ensure_local(self, hub_id: str, local_dir: Path) -> None:
        """Ensure model exists at local_dir; download+save if missing."""
        local_dir = Path(local_dir)
        if (local_dir / "config.json").exists():
            return  # Model already cached locally

        # Try to download if not present
        try:
            tokenizer = AutoTokenizer.from_pretrained(hub_id, local_files_only=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(hub_id, local_files_only=False)
            local_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(local_dir)
            model.save_pretrained(local_dir)
        except Exception as e:
            raise RuntimeError(
                f"Could not find local model at {local_dir} and failed to download.\n"
                "If you are offline, please connect to the internet once so the model can be downloaded, "
                f"then it will be used locally afterwards.\nOriginal error: {e}"
            )

    def get_pipeline(self, key: str):
        """Return (pipeline, needs_prefix) for selected key."""
        if key in self.loaded:
            return self.loaded[key]

        spec = SUPPORTED_MODELS[key]
        hub_id = spec["hub_id"]
        local_dir = spec["local_dir"]
        needs_prefix = spec["needs_prefix"]

        # Ensure local copy exists (download once)
        self._ensure_local(hub_id, local_dir)

        # Load strictly from local to guarantee offline behavior later
        tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(local_dir, local_files_only=True)

        pipe = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=self.device,
        )
        self.loaded[key] = (pipe, needs_prefix)
        return self.loaded[key]


class SummarizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x720")
        self.minsize(900, 600)

        self.model_mgr = ModelManager()
        self.task_queue = queue.Queue()
        self._build_ui()

    # -----------------------------
    # UI Setup
    # -----------------------------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)

        # Top controls frame
        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Model:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value=list(SUPPORTED_MODELS.keys())[0])
        self.model_combo = ttk.Combobox(
            top, textvariable=self.model_var, values=list(SUPPORTED_MODELS.keys()), state="readonly"
        )
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=(6, 12))

        # Params
        params = ttk.Frame(top)
        params.grid(row=0, column=2, sticky="e")

        self.max_len_var = tk.IntVar(value=180)
        self.min_len_var = tk.IntVar(value=30)
        self.beam_var = tk.IntVar(value=4)
        self.sample_var = tk.BooleanVar(value=False)

        def add_spin(label, var, frm, from_, to, width=6):
            f = ttk.Frame(frm)
            ttk.Label(f, text=label).pack(side=tk.LEFT)
            sp = ttk.Spinbox(f, from_=from_, to=to, textvariable=var, width=width)
            sp.pack(side=tk.LEFT, padx=(4, 10))
            f.pack(side=tk.LEFT)

        add_spin("Max:", self.max_len_var, params, 32, 1024)
        add_spin("Min:", self.min_len_var, params, 8, 512)
        add_spin("Beams:", self.beam_var, params, 1, 16)
        ttk.Checkbutton(params, text="Sample", variable=self.sample_var).pack(side=tk.LEFT, padx=(10, 0))

        # Buttons
        btns = ttk.Frame(self, padding=(10, 0))
        btns.grid(row=1, column=0, sticky="ew")
        btns.columnconfigure(4, weight=1)

        self.summarize_btn = ttk.Button(btns, text="Summarize", command=self.on_summarize)
        self.clear_btn = ttk.Button(btns, text="Clear", command=self.on_clear)
        self.load_btn = ttk.Button(btns, text="Load File…", command=self.on_load_file)
        self.save_btn = ttk.Button(btns, text="Save Summary…", command=self.on_save_summary)

        self.summarize_btn.grid(row=0, column=0, padx=5)
        self.clear_btn.grid(row=0, column=1, padx=5)
        self.load_btn.grid(row=0, column=2, padx=5)
        self.save_btn.grid(row=0, column=3, padx=5)

        self.status_var = tk.StringVar(value="Ready")
        self.status = ttk.Label(btns, textvariable=self.status_var)
        self.status.grid(row=0, column=4, sticky="e")

        # Input/Output panes
        io = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        io.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

        # Input frame
        input_frame = ttk.Frame(io)
        input_frame.grid_rowconfigure(1, weight=1)
        input_frame.grid_columnconfigure(0, weight=1)

        ttk.Label(input_frame, text="Input Text").grid(row=0, column=0, sticky="w")

        self.input_txt = tk.Text(input_frame, wrap=tk.WORD, undo=True)
        input_scroll = ttk.Scrollbar(input_frame, command=self.input_txt.yview)
        self.input_txt.configure(yscrollcommand=input_scroll.set)

        self.input_txt.grid(row=1, column=0, sticky="nsew")
        input_scroll.grid(row=1, column=1, sticky="ns")

        # Output frame
        output_frame = ttk.Frame(io)
        output_frame.grid_rowconfigure(1, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)

        ttk.Label(output_frame, text="Summary").grid(row=0, column=0, sticky="w")

        self.output_txt = tk.Text(output_frame, wrap=tk.WORD, state=tk.NORMAL)
        output_scroll = ttk.Scrollbar(output_frame, command=self.output_txt.yview)
        self.output_txt.configure(yscrollcommand=output_scroll.set)

        self.output_txt.grid(row=1, column=0, sticky="nsew")
        output_scroll.grid(row=1, column=1, sticky="ns")

        # Add to paned window
        io.add(input_frame, weight=1)
        io.add(output_frame, weight=1)

        # Periodically process results from worker thread
        self.after(100, self._poll_queue)


    # -----------------------------
    # Actions
    # -----------------------------
    def on_clear(self):
        self.input_txt.delete("1.0", tk.END)
        self.output_txt.delete("1.0", tk.END)
        self.status_var.set("Cleared")

    def on_load_file(self):
        path = filedialog.askopenfilename(
            title="Open text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self.input_txt.delete("1.0", tk.END)
            self.input_txt.insert(tk.END, content)
            self.status_var.set(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def on_save_summary(self):
        content = self.output_txt.get("1.0", tk.END).strip()
        if not content:
            messagebox.showinfo("Info", "There is no summary to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save summary as",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            self.status_var.set(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def on_summarize(self):
        text = self.input_txt.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Info", "Please paste or load some text first.")
            return

        model_key = self.model_var.get()
        params = SummarizeParams(
            max_length=int(self.max_len_var.get()),
            min_length=int(self.min_len_var.get()),
            num_beams=int(self.beam_var.get()),
            do_sample=bool(self.sample_var.get()),
        )

        self.status_var.set(f"Loading model and summarizing with {model_key}…")
        self.summarize_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._worker_summarize, args=(model_key, text, params), daemon=True)
        thread.start()

    def _worker_summarize(self, model_key: str, text: str, params: SummarizeParams):
        try:
            pipe, needs_prefix = self.model_mgr.get_pipeline(model_key)

            if needs_prefix:
                # T5 convention
                text = f"summarize: {text}"

            # Safety guards for extreme inputs
            max_len = max(8, min(1024, params.max_length))
            min_len = max(1, min(max_len - 1, params.min_length))
            num_beams = max(1, min(16, params.num_beams))

            out = pipe(
                text,
                max_length=max_len,
                min_length=min_len,
                num_beams=num_beams,
                do_sample=params.do_sample,
                truncation=True,
            )

            summary = out[0]["summary_text"].strip()
            self.task_queue.put(("ok", summary))
        except Exception as e:
            self.task_queue.put(("err", str(e)))

    def _poll_queue(self):
        try:
            status, payload = self.task_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_queue)
            return

        if status == "ok":
            self.output_txt.delete("1.0", tk.END)
            self.output_txt.insert(tk.END, payload)
            self.status_var.set("Done")
        else:
            messagebox.showerror("Error", payload)
            self.status_var.set("Error")

        self.summarize_btn.config(state=tk.NORMAL)
        self.after(100, self._poll_queue)


if __name__ == "__main__":
    try:
        app = SummarizerApp()
        app.mainloop()
    except KeyboardInterrupt:
        sys.exit(0)
