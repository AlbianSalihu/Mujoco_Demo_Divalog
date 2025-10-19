# src/utils.py
import numpy as np
import mujoco

def get_body_pos(model, data, name):
    """Retourne la position monde (x, y, z) d'un body par son nom."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid == -1:
        raise ValueError(f"Body '{name}' introuvable dans le modèle.")
    return data.xpos[bid].copy()

def compute_desired_orientation_2(model, data, ee_id, screw_body_id=None):
    """
    Orientation cible :
      - l’outil pointe vers le bas (z_d = [0,0,-1])
      - si screw_body_id est fourni : applique une rotation en Z pour aligner
        l'axe de la pince avec l'orientation de la vis sur le plan XY.
    """
    R_curr = data.xmat[ee_id].reshape(3, 3)
    z_d = np.array([0.0, 0.0, -1.0])        # Z outil vers le bas

    # Base : on garde un X proche de l’actuel
    x_guess = R_curr[:, 0]
    x_d = x_guess - z_d * (x_guess @ z_d)
    if np.linalg.norm(x_d) < 1e-9:
        x_d = np.array([1.0, 0.0, 0.0]) - z_d * (np.array([1.0, 0.0, 0.0]) @ z_d)
    x_d /= np.linalg.norm(x_d)

    # ----- YAW ALIGNEMENT avec la vis -----
    if screw_body_id is not None:
        # axe principal de la vis dans le monde (local z -> monde)
        R_screw = data.xmat[screw_body_id].reshape(3, 3)
        screw_axis = R_screw[:, 2]

        # projeté sur le plan XY
        dir_xy = np.array([screw_axis[0], screw_axis[1], 0.0])
        if np.linalg.norm(dir_xy) > 1e-6:
            dir_xy /= np.linalg.norm(dir_xy)
            # angle entre X outil et axe de la vis sur XY
            yaw = np.arctan2(dir_xy[1], dir_xy[0])
            # rotation en Z (autour de z_d)
            Rz = np.array([
                [ np.cos(yaw), -np.sin(yaw), 0],
                [ np.sin(yaw),  np.cos(yaw), 0],
                [ 0, 0, 1]
            ])
            x_d = Rz @ x_d  # on tourne l’axe X dans le plan XY

    y_d = np.cross(z_d, x_d)
    return np.column_stack([x_d, y_d, z_d])

def compute_desired_orientation(model, data, ee_id):
    """Construit une rotation cible orientant le TCP vers le bas."""
    R_curr = data.xmat[ee_id].reshape(3, 3)
    z_d = np.array([0.0, 0.0, -1.0])           # Z vers le bas
    x_guess = R_curr[:, 0]                     # X proche de l'actuel
    x_d = x_guess - z_d * (x_guess @ z_d)
    if np.linalg.norm(x_d) < 1e-9:
        x_d = np.array([1.0, 0.0, 0.0]) - z_d * (np.array([1.0,0.0,0.0]) @ z_d)
    x_d /= np.linalg.norm(x_d)
    y_d = np.cross(z_d, x_d)
    return np.column_stack([x_d, y_d, z_d])