"""
Theme and styling definitions for BioSage Terminal.
Based on the comprehensive styling scheme specification.
"""

from textual.design import ColorSystem


# Color palette based on the styling scheme
COLORS = {
    # Primary Colors
    "primary": "#3B82F6",
    "primary_dark": "#2563EB",
    "primary_light": "#93C5FD",
    
    # Semantic Colors
    "success": "#22C55E",
    "warning": "#EAB308",
    "error": "#EF4444",
    "info": "#06B6D4",
    
    # Neutral Scale
    "surface": "#0F172A",
    "panel": "#1E293B",
    "border": "#334155",
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    
    # Accent Colors
    "accent_purple": "#A855F7",
    "accent_orange": "#F97316",
    "accent_teal": "#14B8A6",
}


# Custom color system for Textual
BIOSAGE_COLOR_SYSTEM = ColorSystem(
    primary=COLORS["primary"],
    secondary=COLORS["panel"],
    warning=COLORS["warning"],
    error=COLORS["error"],
    success=COLORS["success"],
    accent=COLORS["info"],
    dark=True,
    background=COLORS["surface"],
    surface=COLORS["panel"],
    panel=COLORS["panel"],
)


# Specialty colors for agent cards
SPECIALTY_COLORS = {
    "infectious": "#EF4444",
    "cardiology": "#F97316",
    "neurology": "#A855F7",
    "oncology": "#EC4899",
    "autoimmune": "#EAB308",
    "toxicology": "#22C55E",
}


# Priority colors
PRIORITY_COLORS = {
    "low": COLORS["success"],
    "medium": COLORS["warning"],
    "high": COLORS["accent_orange"],
    "critical": COLORS["error"],
}


# Status colors
STATUS_COLORS = {
    "online": COLORS["success"],
    "degraded": COLORS["warning"],
    "offline": COLORS["error"],
}


# Box drawing characters for borders
BOX_CHARS = {
    # Standard single-line
    "single": {
        "top_left": "+",
        "top_right": "+",
        "bottom_left": "+",
        "bottom_right": "+",
        "horizontal": "-",
        "vertical": "|",
        "t_down": "+",
        "t_up": "+",
        "t_right": "+",
        "t_left": "+",
        "cross": "+",
    },
    # Rounded corners
    "rounded": {
        "top_left": "+",
        "top_right": "+",
        "bottom_left": "+",
        "bottom_right": "+",
        "horizontal": "-",
        "vertical": "|",
    },
    # Double-line
    "double": {
        "top_left": "+",
        "top_right": "+",
        "bottom_left": "+",
        "bottom_right": "+",
        "horizontal": "=",
        "vertical": "|",
    },
}


# Tree view characters
TREE_CHARS = {
    "branch": "|--",
    "last_branch": "`--",
    "pipe": "|  ",
    "space": "   ",
    "expand": "[+]",
    "collapse": "[-]",
}


# Progress bar characters
PROGRESS_CHARS = {
    "fill": "#",
    "empty": "-",
    "left_cap": "[",
    "right_cap": "]",
}


# Spinner frames for loading animation
SPINNER_FRAMES = [
    "|", "/", "-", "\\",
]


# Text icons for specialties (no emojis)
SPECIALTY_ICONS = {
    "infectious": "[INF]",
    "cardiology": "[CAR]",
    "neurology": "[NEU]",
    "oncology": "[ONC]",
    "autoimmune": "[AUT]",
    "toxicology": "[TOX]",
}


# Status indicators (no emojis)
STATUS_INDICATORS = {
    "online": "[ON]",
    "degraded": "[DG]",
    "offline": "[--]",
    "success": "[OK]",
    "warning": "[!!]",
    "error": "[XX]",
    "info": "[ii]",
}


# Key binding hints
KEY_HINTS = {
    "dashboard": "[d] Dashboard",
    "onboarding": "[o] Onboarding",
    "specialists": "[s] Specialists",
    "evidence": "[e] Evidence",
    "quit": "[q] Quit",
    "back": "[Esc] Back",
    "help": "[?] Help",
}


def get_priority_color(priority: str) -> str:
    """Get the color for a priority level."""
    return PRIORITY_COLORS.get(priority.lower(), COLORS["text_secondary"])


def get_status_color(status: str) -> str:
    """Get the color for a status."""
    return STATUS_COLORS.get(status.lower(), COLORS["text_muted"])


def get_specialty_color(specialty: str) -> str:
    """Get the color for a specialty."""
    return SPECIALTY_COLORS.get(specialty.lower(), COLORS["primary"])


def format_progress_bar(
    value: float,
    width: int = 20,
    fill_char: str = "#",
    empty_char: str = "-",
) -> str:
    """Format a progress bar string."""
    filled = int(value * width)
    empty = width - filled
    return f"[{fill_char * filled}{empty_char * empty}]"


def format_confidence_bar(confidence: float, width: int = 20) -> str:
    """Format a confidence score as a progress bar."""
    percentage = int(confidence * 100)
    return f"{format_progress_bar(confidence, width)} {percentage}%"


def format_header(title: str, width: int = 60, char: str = "=") -> str:
    """Format a header with decorative borders."""
    padding = (width - len(title) - 2) // 2
    return f"{char * padding} {title} {char * padding}"


def format_box(
    content: list[str],
    width: int = 40,
    title: str = "",
    style: str = "single",
) -> list[str]:
    """Format content in a box with borders."""
    chars = BOX_CHARS[style]
    lines = []
    
    # Top border
    if title:
        title_display = f" {title} "
        border_left = chars["horizontal"] * 2
        border_right = chars["horizontal"] * (width - len(title_display) - 4)
        lines.append(f"{chars['top_left']}{border_left}{title_display}{border_right}{chars['top_right']}")
    else:
        lines.append(f"{chars['top_left']}{chars['horizontal'] * (width - 2)}{chars['top_right']}")
    
    # Content
    for line in content:
        padded = line.ljust(width - 4)[:width - 4]
        lines.append(f"{chars['vertical']} {padded} {chars['vertical']}")
    
    # Bottom border
    lines.append(f"{chars['bottom_left']}{chars['horizontal'] * (width - 2)}{chars['bottom_right']}")
    
    return lines


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to a maximum length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
