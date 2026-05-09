"""
==============================================
KONFIGURATIONSDATEI FÜR WIRE HARNESS ROUTING
==============================================
Diese Datei enthält alle zentralen Parameter und Einstellungen für die
Multi-Agent Wire Harness Routing Simulation.

STRUKTUR:
1. Hyperparameter für Reinforcement Learning
2. Simulationsparameter
3. Mover-Konfigurationen
4. Kabel-Verbindungen
5. Zielkonfigurationen
6. Dateipfade

MOVER-ZUORDNUNG:
Index 0: RED     --> Modul KS_MIKO_17RT      --> Platform P1 --> Kabel 1,2,3,4
Index 1: GREEN   --> Modul TEMPMANNHILI_LWB  --> Platform P2 --> Kabel 1
Index 2: YELLOW  --> Modul SCHALTBET_B       --> Platform P3 --> Kabel 2
Index 3: PURPLE  --> Modul SCHALTBET_A       --> Platform P6 --> Kabel 3
Index 4: ORANGE  --> Modul TEMPMANNHIRE_LWB  --> Platform P7 --> Kabel 4
==============================================
"""

import os

# =============================================================================
# REINFORCEMENT LEARNING HYPERPARAMETER
# =============================================================================

# Deep Q-Network Parameter
EPSILON = 1.0                   # Initiale Exploration-Rate (1.0 = 100% zufällige Aktionen)
EPSILON_DECAY = 0.99999        # Decay-Faktor für Epsilon nach jeder Episode
EPSILON_MIN = 0.01              # Minimaler Epsilon-Wert (1% Exploration bleibt)
GAMMA = 0.99                    # Discount-Faktor für zukünftige Rewards
LR = 0.001                      # Learning Rate für neuronales Netzwerk
BATCH_SIZE = 64                 # Batch-Größe für Experience Replay
BUFFER_SIZE = 500000            # Maximale Größe des Replay Buffers
STATE_SIZE = 31                 # Dimension des State-Vektors (5 Mover * 6 Features + 1 Targetnummer)
ACTION_SIZE = 10                # Anzahl möglicher diskretisierter Aktionen

# Trainingsparameter
NUM_EPISODES = 100000           # Maximale Anzahl von Trainingsepisoden
MAX_TRAJECTORY_LENGTH = 250    # Maximale Schritte pro Episode

# Reward-Shaping Parameter
ALPHA = 2                       # Gewichtung für Distanz-basierte Rewards
BETA = 0.025                    # Gewichtung für Winkel-basierte Rewards
MIN_DIST = 0.35                 # Minimaler erlaubter Abstand zwischen Movern [m]

# =============================================================================
# SIMULATIONSPARAMETER
# =============================================================================

NUM_MOVERS = 5                  # Anzahl der Mover/Agenten im System
SIMEND = 30                     # Maximale Simulationszeit [s]
TABLE_SIZE = 3.5                # Größe des Arbeitsbereichs [m]
VEL = 2                         # Basis-Geschwindigkeit der Mover [m/s]
START_SEQUENCE = 10             # Verzögerung vor Bewegungsstart [Zeitschritte]

# Physik-Simulation
PHYSICS_TIMESTEP = 1.0/60.0     # Zeitschritt für MuJoCo Physics Engine [s]
VIDEO_FPS = 30                  # Framerate für Video-Export

# Grid-Dimensionen für Collision Maps
GRID_WIDTH = 72                 # Breite des Diskretisierungsgitters
GRID_HEIGHT = 50                # Höhe des Diskretisierungsgitters
GRID_RESOLUTION = 0.1           # Auflösung: 1 Grid-Zelle = 0.1m

# =============================================================================
# MOVER-KONFIGURATIONEN
# =============================================================================

# Startpositionen der 5 Mover [x, y] in Metern
# Reihenfolge: [RED, GREEN, YELLOW, PURPLE, ORANGE]
MU_START = [
    [4.0, 3.0],    # RED (P1) - oben rechts
    [5.0, 2.0],    # GREEN (P2) - mitte rechts
    [1.0, 2.3],    # YELLOW (P3) - mitte links
    [1.5, 1.8],    # PURPLE (P6) - mitte links unten
    [4.0, 0.5]     # ORANGE (P7) - unten rechts
]

# MuJoCo Joint-Namen für X/Y-Bewegung
JOINT_NAMES = [
    "slide_joint1",   # RED
    "slide_joint2",   # GREEN
    "slide_joint3",   # YELLOW
    "slide_joint6",   # PURPLE
    "slide_joint7"    # ORANGE
]

# Body-IDs in MuJoCo für die Mover-Plattformen
BODY_IDS = [91, 99, 105, 110, 115]

# Initiale Bewegungsrichtungen beim Start [x, y]
MU_START_MOVE = [
    [0, 0],         # RED - keine initiale Bewegung
    [-0.5, 0],      # GREEN - nach links
    [0.1, 0.5],     # YELLOW - leicht rechts oben
    [-0.1, 0.4],    # PURPLE - leicht links oben
    [-0.3, 0.3]     # ORANGE - diagonal links oben
]

# Follow-Flags: True = folgt Mover 0 (RED), False = unabhängig
MU_FOLLOW = [False, True, True, True, True]

# Maximale erlaubte Abstände zu anderen Movern [m]
# 99 = unbegrenzt (für den führenden Mover)
MAX_DIST = [99, 1.3, 3.1, 2.7, 2.4]

# =============================================================================
# KABEL-KONFIGURATIONEN
# =============================================================================

# Welche Kabel sind mit welchem Mover verbunden
# Kabel-IDs: 1=Wire1, 2=Wire2, 3=Wire3, 4=Wire4
CABLE_CONNECT = [
    [1, 2, 3, 4],   # RED - alle 4 Kabel
    [1],            # GREEN - nur Wire1
    [2],            # YELLOW - nur Wire2
    [3],            # PURPLE - nur Wire3
    [4]             # ORANGE - nur Wire4
]

# Body-IDs der Kabel-Startpunkte für jeden Mover
# Diese Bodies werden in der Collision-Detection speziell behandelt
CABLE_START_MU = [
    [1, 2, 3, 38, 39, 40, 68, 69, 70, 71, 72, 73],  # RED
    [8, 9, 10],                                       # GREEN
    [11, 12, 13],                                     # YELLOW
    [41, 42, 43],                                     # PURPLE
    [88, 89, 90]                                      # ORANGE
]

CABLE_START = [1, 2, 3, 8, 9, 10, 11, 12, 13, 38, 39, 40, 41, 42, 43, 68, 69, 70, 71, 72, 73, 88, 89, 90]

# Body-IDs der Kabel-Endpunkte
CABLE_END = [10, 40, 70, 90]

# Nachbarschaftsbeziehungen für koordinierte Bewegung
# Index des nächsten Nachbarn in der Kette
NEIGHBOR = [1, 4, 3, 2, 1]

# =============================================================================
# ZIELKONFIGURATIONEN
# =============================================================================

# Alle möglichen Zielkonfigurationen
# Format: [Konfiguration][Mover][x,y]
# Jede Konfiguration definiert Zielpositionen für alle 5 Mover
MOVER_TARGETS = [
    # Konfiguration 0: Rechts oben Cluster
    [[5.2, 2.8], [5.3, 1.6], [2.5, 2.8], [3.1, 1.5], [5.1, 0.5]],
    
    # Konfiguration 1: Rechts unten Cluster  
    [[5.2, 0.6], [4.0, 0.3], [5.4, 3.4], [5.0, 3.0], [2.8, 1.1]],
    
    # Konfiguration 2: Mitte Cluster
    [[3.0, 2.8], [3.5, 1.6], [0.5, 2.8], [1.1, 1.5], [3.1, 0.4]],
    
    # Konfiguration 3: Links Cluster
    [[3.2, 0.4], [2.0, 0.3], [3.4, 3.4], [3.0, 3.0], [0.8, 0.7]],
    
    # Konfiguration 4: Finale Positionen (Taping-Stationen)
    [[1.2, 0.8], [1.0, 2.0], [4.2, 0.4], [3.6, 0.8], [1.4, 3.2]]
]

# Sequenz der Zielkonfigurationen (welche Configs nacheinander angefahren werden)
# [4,2,3,1,0] bedeutet: Config 4 -> Config 2 -> Config 3 -> Config 1 -> Config 0
TARGET_SEQUENCE = [4, 2, 3, 1, 0]

# =============================================================================
# STATIONEN UND WEGPUNKTE
# =============================================================================

# Spezielle Stationen im Arbeitsbereich (aus XML)
STATIONS = {
    "green_sphere": [1.0, 2.0, 0],      # Grüne Zielmarkierung
    "pick_up_station": [1.4, 3.2, 0],   # Orange Aufnahmestation
    "taping_station1": [1.2, 2.0, 0.3], # Blaue Taping-Station (erhöht)
    "taping_station2": [1.2, 0.8, 0],   # Rote Taping-Station
    "taping_station3": [4.2, 0.4, 0],   # Gelbe Taping-Station
    "test_sphere": [3.6, 0.8, 0]        # Lila Test-Station
}

# =============================================================================
# DATEIPFADE UND VERZEICHNISSE
# =============================================================================

# Projektverzeichnis (Basis für alle relativen Pfade)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# MuJoCo XML-Modell
DATA_DIR = os.path.join(PROJECT_DIR, "data")               # Daten und Configs
XML_PATH = os.path.join(DATA_DIR, "WireHarness012e.xml")

# Ausgabeverzeichnisse
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "rlagents", "checkpoints")  # Modell-Checkpoints
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
VIDEO_DIR = os.path.join(RESULTS_DIR, "videos")           # Video-Ausgaben
LOG_DIR = os.path.join(RESULTS_DIR, "logs")                # Trainings-Logs
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")              # Visualisierungen
STATISTICS_DIR = os.path.join(RESULTS_DIR, "statistics")              # Visualisierungen
TENSORBOARD_DIR = os.path.join(RESULTS_DIR, "tensorboard")            # TensorBoard-Logs

# Simulationskonfigurationen
SIM_CONFIG_DIR = os.path.join(DATA_DIR, "simulation_config")
BODY_CONFIG_FILE = os.path.join(SIM_CONFIG_DIR, "body_012b.xml")
GEOM_CONFIG_FILE = os.path.join(SIM_CONFIG_DIR, "geom_007.xml")

# CSV-Ausgaben
TRAJECTORY_FILE = os.path.join(DATA_DIR, "best_trajectory.csv")

# =============================================================================
# RENDERING UND VISUALISIERUNG
# =============================================================================

# Video-Einstellungen
VIDEO_WIDTH = 640               # Video-Breite in Pixeln
VIDEO_HEIGHT = 352              # Video-Höhe in Pixeln
VIDEO_CODEC = "libx264"         # Video-Codec für MP4
VIDEO_QUALITY = 1               # Qualität (1 = beste)

# Kamera-Einstellungen (MuJoCo)
CAMERA_AZIMUTH = 90.0           # Horizontale Rotation [Grad]
CAMERA_DISTANCE = 4.5           # Zoom-Distanz [m]
CAMERA_ELEVATION = -60.0        # Vertikale Rotation [Grad]
CAMERA_LOOKAT = [3.36, 1.6, 0.0]  # Fokuspunkt [x, y, z]

# Farben für Visualisierung (RGBA)
COLORS = {
    "red": [1, 0, 0, 1],
    "green": [0, 1, 0, 1],
    "yellow": [1, 1, 0, 1],
    "purple": [1, 0, 1, 1],
    "orange": [1, 0.5, 0, 1],
    "blue": [0, 0, 1, 1],
    "gray": [0.5, 0.5, 0.5, 1]
}

# =============================================================================
# DEBUG UND ENTWICKLUNG
# =============================================================================

DEBUG_MODE = False              # Aktiviert zusätzliche Debug-Ausgaben
SAVE_COLLISION_MAPS = False    # Speichert Collision-Maps als Bilder
VERBOSE_LOGGING = False         # Detaillierte Console-Ausgaben
PROFILE_PERFORMANCE = False     # Performance-Profiling aktivieren

# Headless Mode (ohne GUI)
HEADLESS = True                 # True für Server, False für Desktop
MUJOCO_GL_BACKEND = "glfw" if not HEADLESS else "egl"  # Rendering-Backend