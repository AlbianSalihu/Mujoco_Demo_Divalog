# src/sim.py
import os
import mujoco
from config import PANDA_DIR, PANDA_FILE, SCENE_XML, OVERHEAD_NAME

def build_model():
    """Charge la scène, écrit un XML temporaire dans PANDA_DIR, renvoie (model, data, over_id)."""
    with open(SCENE_XML, "r") as f:
        xml_text = f.read().replace("{PANDA_PATH}", PANDA_FILE)

    tmp_scene_path = os.path.join(PANDA_DIR, "_scene_tmp.xml")
    with open(tmp_scene_path, "w") as f:
        f.write(xml_text)

    model = mujoco.MjModel.from_xml_path(tmp_scene_path)
    data  = mujoco.MjData(model)

    over_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, OVERHEAD_NAME)
    if over_id == -1:
        raise ValueError(f"Camera '{OVERHEAD_NAME}' introuvable dans le XML.")

    return model, data, over_id
