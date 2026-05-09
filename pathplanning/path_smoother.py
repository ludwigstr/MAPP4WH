"""
==============================================
PATH SMOOTHER - Glättet Pfade für professionelle Visualisierung
==============================================
"""

import numpy as np
from scipy.interpolate import CubicSpline


def smooth_path_cubic_spline(path, num_points=50):
    """
    Glättet einen Pfad mit Cubic Spline Interpolation.
    
    Args:
        path: Liste von (x, y) Grid-Koordinaten Tupeln
        num_points: Anzahl der Punkte im geglätteten Pfad
    
    Returns:
        Geglätteter Pfad als Liste von (x, y) in Metern
    """
    if len(path) < 3:
        # Zu kurz für Smoothing, gib Original zurück
        return [(p[0] / 10.0, p[1] / 10.0) for p in path]
    
    # Extrahiere x und y Koordinaten (Grid zu Meter)
    path_x = np.array([p[0] / 10.0 for p in path])
    path_y = np.array([p[1] / 10.0 for p in path])
    
    # Parameter t (0 bis 1) für jeden Pfadpunkt
    t = np.linspace(0, 1, len(path))
    
    # Neue Punkte für geglätteten Pfad
    t_smooth = np.linspace(0, 1, num_points)
    
    try:
        # Cubic Spline für x und y Koordinaten
        cs_x = CubicSpline(t, path_x, bc_type='natural')
        cs_y = CubicSpline(t, path_y, bc_type='natural')
        
        # Geglättete Koordinaten berechnen
        smooth_x = cs_x(t_smooth)
        smooth_y = cs_y(t_smooth)
        
        # Als Liste von Tupeln zurückgeben
        smooth_path = list(zip(smooth_x, smooth_y))
        
        return smooth_path
    
    except Exception as e:
        # Fallback: Bei Fehler Original-Pfad zurückgeben
        print(f"[Path Smoother] Fehler beim Smoothing: {e}")
        return [(p[0] / 10.0, p[1] / 10.0) for p in path]


def smooth_path_moving_average(path, window_size=5):
    """
    Einfaches Smoothing mit Moving Average (Fallback-Option).
    
    Args:
        path: Liste von (x, y) Grid-Koordinaten Tupeln
        window_size: Größe des Glättungsfensters
    
    Returns:
        Geglätteter Pfad als Liste von (x, y) in Metern
    """
    if len(path) < window_size:
        return [(p[0] / 10.0, p[1] / 10.0) for p in path]
    
    path_x = np.array([p[0] / 10.0 for p in path])
    path_y = np.array([p[1] / 10.0 for p in path])
    
    smooth_x = []
    smooth_y = []
    
    half_window = window_size // 2
    
    for i in range(len(path)):
        start = max(0, i - half_window)
        end = min(len(path), i + half_window + 1)
        
        smooth_x.append(np.mean(path_x[start:end]))
        smooth_y.append(np.mean(path_y[start:end]))
    
    return list(zip(smooth_x, smooth_y))