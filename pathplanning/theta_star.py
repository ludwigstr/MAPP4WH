"""
==============================================
THETA* - ANY-ANGLE PATHFINDING
==============================================
Optimierter Pfadplanungs-Algorithmus für glatte Trajektorien
MIT SOFT CONSTRAINTS: Pfade sind teuer, aber passierbar
MIT PARALLEL-ERKENNUNG: Nur bei direkter Bewegung aktiv
"""

import numpy as np
import heapq
import math
from pathplanning.path_smoother import smooth_path_cubic_spline


def compute_theta_star_action(env, mover, goal_x, goal_y, clockwise, c_x, c_y, use_rotation=True):
    """
    Theta* Pathfinding - Erzeugt glatte Any-Angle Pfade.
    
    Args:
        env: Environment-Objekt
        mover: Mover-Objekt
        goal_x, goal_y: Zielposition
        clockwise: Rotationsrichtung (True/False)
        c_x, c_y: Zentrum für Kreisbahn
        use_rotation: Bool, ob Rotationsgewicht in Heuristik verwendet werden soll
    
    Returns:
        action: [dx, dy] Bewegungsvektor (normalisiert)
    """
    
    # ========== COLLISION MAP VORBEREITUNG ==========
    collision_map_copy = np.copy(env.collision_map)

    # Eigenen Mover entfernen
    for y in range(collision_map_copy.shape[0]):
        for x in range(collision_map_copy.shape[1]):
            if collision_map_copy[y, x] == mover.mu_index:
                collision_map_copy[y, x] = 0

    updated_map = collision_map_copy.copy()

    # Kabelstart-Punkte ausblenden
    for i in mover.cable_start_mu:
        try:
            x, y, _ = env.data.xpos[i]
            x_idx = int(round(x, 1) * 10)
            y_idx = int(round(y, 1) * 10)
            updated_map[y_idx, x_idx] = 0
        except:
            pass

    # Mover-Bereiche erweitern für Sicherheitsabstand
    for y in range(collision_map_copy.shape[0]):
        for x in range(collision_map_copy.shape[1]):
            for i in range(env.num_agents):
                if collision_map_copy[y, x] == env.movers[i].mu_index:
                    neighbors = [
                        (y-1, x-1), (y-1, x), (y-1, x+1),
                        (y, x-1),           (y, x+1),
                        (y+1, x-1), (y+1, x), (y+1, x+1)
                    ]
                    for ny, nx in neighbors:
                        if 0 <= ny < collision_map_copy.shape[0] and 0 <= nx < collision_map_copy.shape[1]:
                            if updated_map[ny, nx] == 0:
                                updated_map[ny, nx] = env.movers[i].mu_index

    collision_map_copy = updated_map.copy()

    # Extra Sicherheit für nahe Mover
    for i in range(env.num_agents):
        dist = mover.get_distance(env.movers[i].x, env.movers[i].y)
        if 0 < dist < 0.6:
            for y in range(collision_map_copy.shape[0]):
                for x in range(collision_map_copy.shape[1]):
                    if collision_map_copy[y, x] == env.movers[i].mu_index:
                        neighbors = [
                            (y-1, x-1), (y-1, x), (y-1, x+1),
                            (y, x-1),           (y, x+1),
                            (y+1, x-1), (y+1, x), (y+1, x+1)
                        ]
                        for ny, nx in neighbors:
                            if 0 <= ny < collision_map_copy.shape[0] and 0 <= nx < collision_map_copy.shape[1]:
                                if updated_map[ny, nx] == 0:
                                    updated_map[ny, nx] = env.movers[i].mu_index

    collision_map_copy = updated_map.copy()

    # ========== PARALLEL-ERKENNUNG (NUR BEI DIREKTER BEWEGUNG) ==========
    # NUR wenn use_rotation=False (direkte Bewegung)
    if not use_rotation:
        mover_direction = np.array([goal_x - mover.x, goal_y - mover.y])
        mover_direction_norm = np.linalg.norm(mover_direction)
        
        if mover_direction_norm > 0.01:
            mover_direction = mover_direction / mover_direction_norm
            
            for i in range(env.num_agents):
                other_mover = env.movers[i]
                
                if other_mover.mu_index == mover.mu_index:
                    continue
                
                other_direction = np.array([other_mover.x_t - other_mover.x, 
                                           other_mover.y_t - other_mover.y])
                other_direction_norm = np.linalg.norm(other_direction)
                
                if other_direction_norm > 0.01:
                    other_direction = other_direction / other_direction_norm
                    
                    dot_product = np.dot(mover_direction, other_direction)
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    angle_degrees = np.degrees(angle)
                    
                    # Wenn Winkel < 30° → Kabel des anderen Movers entfernen
                    if angle_degrees < 30:
                        for cable_id in other_mover.cable_connect:
                            for y in range(collision_map_copy.shape[0]):
                                for x in range(collision_map_copy.shape[1]):
                                    if collision_map_copy[y, x] == cable_id:
                                        collision_map_copy[y, x] = 0

    # COST MAP ERSTELLEN (Soft Constraints)
    cost_map = np.ones((50, 72))  # Basis-Kosten = 1.0 für freie Zellen
    
    for y in range(cost_map.shape[0]):
        for x in range(cost_map.shape[1]):
            if collision_map_copy[y, x] != 0:
                # Echte Hindernisse (Kabel, Mover) = unpassierbar
                cost_map[y, x] = float('inf')
            elif env.path_map[y, x] > 0:
                # Pfade anderer Mover = teuer aber passierbar
                cost_map[y, x] = 50.0

    # ========== HILFSFUNKTIONEN ==========
    
    def line_of_sight(x1, y1, x2, y2):
        """
        Prüft ob direkte Sichtlinie zwischen zwei Punkten existiert.
        Line-of-Sight erlaubt KEINE Pfad-Durchquerung (zu riskant).
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            if x < 0 or x >= 72 or y < 0 or y >= 50:
                return False
            # Blockiere Line-of-Sight durch Hindernisse UND Pfade
            if collision_map_copy[y, x] != 0 or env.path_map[y, x] > 0:
                return False
            
            if x == x2 and y == y2:
                return True
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
            return False
    
    def euclidean_distance(a, b):
        """Euklidische Distanz zwischen zwei Punkten."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def heuristic(a, b, clockwise, c):
        """
        Heuristik mit optionaler Rotation.
        ALTE VERSION: Fixes Rotationsgewicht
        
        Du kannst hier zwischen zwei Varianten wählen:
        - return distance + rotation_weight / 360.0  (Original - sehr schwach)
        - return distance + rotation_weight / 10.0   (Stärker - 36x mehr Gewicht)
        """
        distance = euclidean_distance(a, b)
        
        # Bei direkter Bewegung: Keine Rotation
        if not use_rotation:
            return distance
        
        # Bei Kreisbewegung: Distanz + Rotationsgewicht
        Mx, My = c
        x1, y1 = a
        x2, y2 = b
        
        v1x, v1y = x1 - Mx, y1 - My
        v2x, v2y = x2 - Mx, y2 - My
        dot = v1x*v2x + v1y*v2y
        cross = v1x*v2y - v1y*v2x
        ccw = math.degrees(math.atan2(cross, dot)) % 360.0
        adj = (360.0 - ccw) % 360.0 if clockwise else ccw
        
        # Reduziere Rotationsgewicht wenn nah am Ziel
        if distance < 10:
            rotation_weight = adj * (distance / 10.0)
        else:
            rotation_weight = adj
        
        # ========== HIER KANNST DU WÄHLEN ==========
        # Option 1: Original (sehr schwaches Gewicht)
        # return distance + rotation_weight / 360
        
        # Option 2: Stärkeres Gewicht (kommentiere Zeile oben aus, diese aktivieren)
        return distance + rotation_weight / 10.0
    
    def get_neighbors(node):
        """
        8-er Nachbarschaft für Theta*.
        Gibt jetzt auch Nachbarn zurück die auf Pfaden liegen (nur echte Hindernisse ausgeschlossen).
        """
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        neighbors = []
        
        for dy, dx in directions:
            x2, y2 = node[0] + dx, node[1] + dy
            
            if 0 <= x2 < 72 and 0 <= y2 < 50:
                # Nur echte Hindernisse ausschließen (inf), Pfade sind erlaubt (50.0)
                if cost_map[y2, x2] < float('inf'):
                    neighbors.append((x2, y2))
        
        return neighbors
    
    # ========== FRÜHZEITIGE ZIEL-ANNÄHERUNG ==========
    dist_to_goal = math.sqrt((mover.x - goal_x)**2 + (mover.y - goal_y)**2)
    
    if dist_to_goal < 0.5:
        # Prüfe ob direkter Weg frei ist
        direct_free = True
        steps = max(5, int(dist_to_goal * 20))
        
        for i in range(1, steps):
            t = i / float(steps)
            check_x = mover.x + t * (goal_x - mover.x)
            check_y = mover.y + t * (goal_y - mover.y)
            
            x_idx = int(round(check_x, 1) * 10)
            y_idx = int(round(check_y, 1) * 10)
            
            if x_idx < 0 or x_idx >= 72 or y_idx < 0 or y_idx >= 50:
                direct_free = False
                break
            
            if cost_map[y_idx, x_idx] == float('inf'):
                direct_free = False
                break
        
        if direct_free:
            direct_direction = np.array([goal_x - mover.x, goal_y - mover.y])
            direct_norm = np.linalg.norm(direct_direction)
            
            if direct_norm > 0:
                if dist_to_goal < 0.4:
                    speed_factor = max(0.4, dist_to_goal / 0.4)
                    return (direct_direction / direct_norm) * speed_factor
                else:
                    return direct_direction / direct_norm
    
    # ========== START UND ZIEL ==========
    start_idx = (int(round(mover.x, 1) * 10), int(round(mover.y, 1) * 10))
    goal_idx_raw = (int(round(goal_x, 1) * 10), int(round(goal_y, 1) * 10))
    
    goal_idx = (
        min(71, max(0, goal_idx_raw[0])),
        min(49, max(0, goal_idx_raw[1]))
    )
    
    turn_idx = (int(round(c_x, 1) * 10), int(round(c_y, 1) * 10))
    
    # Ziel-Validierung
    if cost_map[goal_idx[1], goal_idx[0]] == float('inf'):
        found = False
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                test_x = goal_idx[0] + dx
                test_y = goal_idx[1] + dy
                
                if 0 <= test_x < 72 and 0 <= test_y < 50:
                    if cost_map[test_y, test_x] < float('inf'):
                        goal_idx = (test_x, test_y)
                        found = True
                        break
            if found:
                break
    
    # ========== THETA* ALGORITHMUS ==========
    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    
    came_from = {}
    g_score = {start_idx: 0}
    f_score = {start_idx: heuristic(start_idx, goal_idx, clockwise, turn_idx)}
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        # Ziel erreicht?
        if current == goal_idx:
            # Pfad rekonstruieren
            path = []
            node = current
            
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start_idx)
            path.reverse()
            
            # SMOOTHING ANWENDEN
            # Original-Pfad für path_map (Grid-Koordinaten)
            mover.path_original = path
            
            # Geglätteter Pfad für Visualisierung (Meter-Koordinaten)
            mover.path = smooth_path_cubic_spline(path, num_points=50)
            
            # PFAD MIT KREUZ-FORM EINTRAGEN (nutzt Original-Pfad)
            for point in mover.path_original:
                y, x = point[1], point[0]
                
                # Hauptpixel
                env.path_map[y, x] = 7
                
                # Kreuz-Form: Nur 4 direkte Nachbarn (oben, unten, links, rechts)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 50 and 0 <= nx < 72:
                        # Nur leere Zellen überschreiben (keine Kabel/Mover)
                        if env.collision_map[ny, nx] == 0:
                            env.path_map[ny, nx] = 7
            
            # ========== BEWEGUNGSRICHTUNG BERECHNEN ==========
            # Nutzt ORIGINAL-Pfad für Bewegung (präziser)
            if len(mover.path_original) >= 2:
                next_point = mover.path_original[1]
                direction = np.array([
                    next_point[0] / 10.0 - mover.x,
                    next_point[1] / 10.0 - mover.y
                ])
                norm = np.linalg.norm(direction)
                
                if norm > 0:
                    if dist_to_goal < 0.4:
                        speed_factor = max(0.5, dist_to_goal / 0.4)
                        return (direction / norm) * speed_factor
                    else:
                        return direction / norm
            
            return np.zeros(2)
        
        # Nachbarn untersuchen
        for neighbor in get_neighbors(current):
            
            # KOSTEN BERECHNEN (mit Pfad-Penalty)
            base_distance = euclidean_distance(current, neighbor)
            path_cost = cost_map[neighbor[1], neighbor[0]]
            edge_cost = base_distance * path_cost
            
            # Theta* Kernlogik - Line of Sight Check
            if current in came_from:
                parent = came_from[current]
                
                if line_of_sight(parent[0], parent[1], neighbor[0], neighbor[1]):
                    parent_to_neighbor_dist = euclidean_distance(parent, neighbor)
                    parent_to_neighbor_cost = parent_to_neighbor_dist * path_cost
                    tentative_g = g_score[parent] + parent_to_neighbor_cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = parent
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal_idx, clockwise, turn_idx)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        continue
            
            # Standard-Pfad (mit Kosten)
            tentative_g = g_score[current] + edge_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal_idx, clockwise, turn_idx)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # Kein Pfad gefunden
    return np.zeros(2)