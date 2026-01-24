# 🖥️ ARGUS Terminal Sandbox

> A Bloomberg-style retro terminal interface for the ARGUS AI Debate System

<div align="center">

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![Textual](https://img.shields.io/badge/textual-0.47+-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Experience AI debates in a stunning 1980s amber or 1970s green phosphor CRT aesthetic**

</div>

---

## 📖 Table of Contents

- [What is ARGUS Terminal?](#-what-is-argus-terminal)
- [Quick Start](#-quick-start)
- [Keyboard Controls](#-keyboard-controls)
- [Screens Overview](#-screens-overview)
- [For Non-Technical Users](#-for-non-technical-users)
- [For Developers](#-for-developers)
- [Troubleshooting](#-troubleshooting)
- [Building Executable](#-building-executable)

---

## 🎯 What is ARGUS Terminal?

ARGUS Terminal is a **text-based user interface (TUI)** that lets you interact with the ARGUS AI Debate System through your terminal. Think of it like a Bloomberg Terminal for AI - packed with information, keyboard-driven, and designed for power users.

### ✨ Features

| Feature | Description |
|---------|-------------|
| 🎨 **Retro Themes** | Amber (1980s) and Green (1970s) phosphor CRT styles |
| ⚔️ **Debate System** | Run AI debates on any proposition |
| ☁️ **27+ LLM Providers** | OpenAI, Anthropic, Google, Mistral, and more |
| ⚙️ **19 Tools** | Search, code execution, databases, finance |
| 📚 **Knowledge Connectors** | Web, ArXiv, CrossRef integration |
| 📊 **Benchmarks** | TruthfulQA, MMLU, GSM8K evaluation |

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Navigate to the argus directory
cd c:\ingester_ops\argus

# Install with terminal extras
pip install -e ".[terminal]"
```

### Step 2: Run the Terminal

**Option A - Entry Point (Recommended after pip install):**
```bash
argus-sandbox
```

**Option B - Python Module:**
```bash
cd c:\ingester_ops\argus
python -m argus_terminal.argus_terminal
```

**Option C - Direct Script:**
```bash
cd c:\ingester_ops\argus\argus_terminal\argus_terminal
python app.py
```

### Step 3: Navigate!

Once the app is running, press number keys **1-8** to switch screens, or **q** to quit.

---

## ⌨️ Keyboard Controls

### Main Navigation

| Key | Action | Screen |
|-----|--------|--------|
| `1` | Go to Dashboard | Main overview with stats |
| `2` | Go to Debate | Start and monitor AI debates |
| `3` | Go to Providers | Configure 27+ LLM providers |
| `4` | Go to Tools | Browse 19 integrated tools |
| `5` | Go to Knowledge | Manage connectors & retrieval |
| `6` | Go to Benchmark | Run evaluation benchmarks |
| `7` | Go to Settings | Configure themes & defaults |
| `8` | Go to Help | View documentation |

### Alternative Keys (if numbers don't work)

| Key | Action |
|-----|--------|
| `F1` - `F8` | Same as 1-8 (may not work in all terminals) |
| `Escape` | Go back to Dashboard |
| `q` | Quit the application |
| `Ctrl+Q` | Quit (alternative) |

### Within Screens

| Key | Action |
|-----|--------|
| `Tab` | Move focus to next widget |
| `Shift+Tab` | Move focus to previous widget |
| `Enter` | Activate focused button/input |
| `Arrow Keys` | Navigate within lists/trees |
| `Space` | Toggle checkboxes/switches |

---

## 📱 Screens Overview

### 1️⃣ Dashboard (Main)

The home screen showing:
- Quick statistics (Providers, Tools, Embeddings, Connectors)
- Action buttons for common tasks
- Recent activity feed

**How to use:** Press buttons or navigate with Tab/Enter.

---

### 2️⃣ Debate System

The core functionality - run AI debates on any claim.

**How to start a debate:**
1. Press `2` to go to Debate screen
2. Enter your proposition (e.g., "Climate change is primarily caused by human activity")
3. Set prior probability (0.5 = neutral, closer to 1 = likely true, closer to 0 = likely false)
4. Choose domain (e.g., Science, Politics, Health)
5. Click "START DEBATE" or press Enter

**Watch the debate:**
- **Agent Orchestra**: See Moderator, Specialist, Refuter, and Jury agents working
- **C-DAG View**: Visual graph of propositions, evidence, and rebuttals
- **Posterior Gauge**: Final probability verdict
- **Activity Log**: Real-time debate progress

---

### 3️⃣ LLM Providers

Configure AI language model providers.

**Available providers include:**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- Cohere, Mistral, Groq
- And 20+ more!

**How to configure:**
1. Select a provider from the tree
2. Enter your API key in the configuration panel
3. Click "Validate & Save"

---

### 4️⃣ Tools

Browse and execute integrated tools.

**Tool categories:**
- 🔍 **Search**: DuckDuckGo, Wikipedia, ArXiv
- 🌐 **Web**: Scraping, page reading
- 💻 **Code**: Python REPL, shell commands
- 📁 **Files**: Read, write, list directories
- 📊 **Data**: SQL, Pandas, JSON/CSV
- 💹 **Finance**: Stock data, SEC filings

**How to use a tool:**
1. Select a tool from the tree
2. Fill in the parameters (if any)
3. Click "Execute Tool"
4. View results in the output panel

---

### 5️⃣ Knowledge & Connectors

Manage external knowledge sources.

**Connectors:**
- **Web Connector**: Scrape websites (respects robots.txt)
- **ArXiv**: Search academic papers
- **CrossRef**: Find citations and DOIs

**Hybrid Retriever:** Combines BM25 (keyword) + semantic search.

**How to ingest documents:**
1. Enter file path or URL
2. Select document type
3. Click "Ingest Document"

---

### 6️⃣ Benchmarking

Evaluate model performance on standard datasets.

**Available benchmarks:**
- TruthfulQA (truthfulness)
- MMLU (multitask understanding)
- GSM8K (math reasoning)
- HumanEval (code generation)
- HellaSwag (commonsense)

**How to run:**
1. Select a dataset
2. Configure samples and rounds
3. Click "RUN BENCHMARK"

---

### 7️⃣ Settings

Customize the terminal.

**Options:**
- **Theme**: Amber (1980s) or Green (1970s)
- **Display**: Scanlines, glow effects, animations
- **Debate Defaults**: Max rounds, prior, specialists
- **Provider Defaults**: Default LLM and model

---

### 8️⃣ Help

View this documentation in-app.

---

## 👨‍💼 For Non-Technical Users

### What Do I Need?

1. **Python 3.11+** installed on your computer
2. **A terminal** (Command Prompt, PowerShell, or Terminal app)
3. **10 minutes** to set up

### Step-by-Step Setup

1. **Open your terminal** (search for "PowerShell" or "Terminal" on your computer)

2. **Navigate to the project folder:**
   ```
   cd c:\ingester_ops\argus
   ```

3. **Install the application:**
   ```
   pip install -e ".[terminal]"
   ```

4. **Run it:**
   ```
   argus-sandbox
   ```

5. **You're in!** Use number keys 1-8 to explore.

### If Something Goes Wrong

- **Screen looks broken?** Try making your terminal window larger
- **Keys not working?** Try using `q` to quit and restart
- **Colors look wrong?** Check your terminal supports 256 colors

---

## 👨‍💻 For Developers

### Project Structure

```
argus_terminal/
├── argus_terminal/          # Main package
│   ├── __init__.py          # Package init & entry point
│   ├── __main__.py          # Module execution
│   ├── app.py               # Main Textual App class
│   ├── config.py            # Configuration management
│   │
│   ├── themes/              # CSS stylesheets
│   │   ├── base.tcss        # Shared styles
│   │   ├── amber.tcss       # 1980s amber theme
│   │   └── green.tcss       # 1970s green theme
│   │
│   ├── screens/             # Application screens
│   │   ├── main.py          # Dashboard
│   │   ├── debate.py        # Debate system
│   │   ├── providers.py     # LLM configuration
│   │   ├── tools.py         # Tool browser
│   │   ├── knowledge.py     # Connectors
│   │   ├── benchmark.py     # Evaluation
│   │   ├── settings.py      # Configuration
│   │   └── help.py          # Documentation
│   │
│   ├── widgets/             # Custom UI components
│   │   ├── header.py        # Top bar
│   │   ├── footer.py        # Bottom bar
│   │   ├── sidebar.py       # Navigation
│   │   ├── command_input.py # Command line
│   │   ├── agent_status.py  # Agent cards
│   │   ├── cdag_view.py     # Graph visualization
│   │   ├── posterior_gauge.py
│   │   ├── log_viewer.py
│   │   └── progress_panel.py
│   │
│   └── utils/
│       └── argus_bridge.py  # Integration with argus package
│
├── build.py                 # PyInstaller build script
├── argus_sandbox.spec       # PyInstaller configuration
├── pyproject.toml           # Package configuration
└── README.md                # This file
```

### Key Components

| File | Purpose |
|------|---------|
| `app.py` | Main `ArgusTerminalApp` class with MODES and BINDINGS |
| `screens/*.py` | Each screen is a `Screen` subclass |
| `widgets/*.py` | Reusable UI components |
| `themes/*.tcss` | Textual CSS for styling |

### Adding a New Screen

1. Create `screens/myscreen.py`:
   ```python
   from textual.screen import Screen
   from textual.app import ComposeResult
   from textual.widgets import Static
   
   class MyScreen(Screen):
       def compose(self) -> ComposeResult:
           yield Static("Hello from MyScreen!")
   ```

2. Register in `app.py`:
   ```python
   from argus_terminal.screens.myscreen import MyScreen
   
   MODES = {
       ...
       "myscreen": MyScreen,
   }
   ```

3. Add binding in `app.py`:
   ```python
   Binding("9", "switch_mode('myscreen')", "My Screen"),
   ```

### Customizing Themes

Edit `themes/amber.tcss` or `themes/green.tcss`:

```css
/* Change primary color */
$primary: #ff9900;

/* Change background */
$background: #0a0a0a;
```

---

## 🔧 Troubleshooting

### "Module not found" Error

```bash
pip install -e ".[terminal]"
```

### "No module named argus_terminal" when running app.py

Run from the correct directory:
```bash
cd c:\ingester_ops\argus
python -m argus_terminal.argus_terminal
```

### Keys Not Working

**Windows Terminal/PowerShell:**
- Try number keys `1-8` instead of F-keys
- Press `q` to quit
- Make sure the terminal window has focus

**VS Code Terminal:**
- F-keys may be captured by VS Code
- Use number keys `1-8` instead

### Screen Looks Broken

1. Make your terminal window larger (at least 120x40)
2. Use a terminal that supports Unicode (e.g., Windows Terminal)
3. Use a font that supports box-drawing characters (e.g., Cascadia Code)

### App Crashes on Startup

Check for Python version:
```bash
python --version  # Should be 3.11 or higher
```

---

## 📦 Building Executable

Create a standalone `.exe` that doesn't require Python:

### Step 1: Install PyInstaller

```bash
pip install pyinstaller
```

### Step 2: Build

```bash
cd c:\ingester_ops\argus\argus_terminal
python build.py
```

Or use the spec file directly:
```bash
pyinstaller argus_sandbox.spec
```

### Step 3: Find Your Executable

The `.exe` will be at:
```
c:\ingester_ops\argus\argus_terminal\dist\argus_sandbox.exe
```

### Distribution

You can now share `argus_sandbox.exe` with anyone - they don't need Python installed!

---

## 📄 License

MIT License - See the main ARGUS repository for details.

---

## 🙏 Acknowledgments

Built with:
- [Textual](https://textual.textualize.io/) - Modern TUI framework
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal formatting
- [ARGUS](https://github.com/Ronit26Mehta/argus-ai-debate) - AI Debate System

---

<div align="center">

**Made with ❤️ for the ARGUS Project**

*Experience the debate. Trust the evidence.*

</div>
