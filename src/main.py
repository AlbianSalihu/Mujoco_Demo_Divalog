"""
Visual servoing controller for Panda robot arm tracking a screw.

Uses overhead camera feedback and inverse kinematics to follow a target object.
"""
import logging
import cv2
import mujoco
import numpy as np
from mujoco import viewer
from typing import Optional, Tuple
from dataclasses import dataclass

from config import H, W
from sim import build_model
from vision import create_renderer, get_rgb_depth, detect_screw_center_bgr
from ik_control import PandaIKController
from utils import compute_desired_orientation_2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ControlConfig:
    """Configuration for visual servoing controller."""
    home_q: np.ndarray = None
    follow_height: float = 0.2
    smooth_alpha: float = 0.2
    max_velocity: float = 0.5
    orientation_weight_max: float = 0.3
    orientation_ramp_distance: float = 0.12
    control_frequency: float = 100.0  # Hz
    physics_substeps: int = 2
    
    def __post_init__(self):
        if self.home_q is None:
            self.home_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


class VisualServoController:
    """Main controller for visual servoing task."""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, 
                 camera_id: int, config: Optional[ControlConfig] = None):
        self.model = model
        self.data = data
        self.config = config or ControlConfig()
        self.camera_id = camera_id
        
        # Initialize IK controller
        self.ik_controller = PandaIKController(model)
        
        # Get body IDs
        self.screw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screw_static")
        self.marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_target_marker")
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        
        if any(id == -1 for id in [self.screw_id, self.marker_id, self.ee_id]):
            raise ValueError("Required bodies not found in model")
        
        self.mocap_id = int(model.body_mocapid[self.marker_id])
        
        # Initialize renderer
        self.renderer = create_renderer(model, H, W)
        
        # State
        self.target_world = np.zeros(3)
        self.running = False
        self.reset()
        
        logger.info("Visual servo controller initialized")
    
    def reset(self):
        """Reset robot to home position."""
        self.data.qpos[:7] = self.config.home_q
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        # Initialize target at current marker position
        self.target_world = self.data.xpos[self.marker_id].copy()
        self.running = False
        
        logger.info("Controller reset to home position")
    
    def _update_target(self) -> np.ndarray:
        """
        Update target position based on screw location.
        Applies exponential smoothing for stability.
        
        Returns:
            Smoothed target position (3,)
        """
        screw_pos = self.data.xpos[self.screw_id].copy()
        target_raw = np.array([
            screw_pos[0],
            screw_pos[1],
            self.config.follow_height
        ])
        
        # Exponential smoothing
        alpha = self.config.smooth_alpha
        self.target_world = (1 - alpha) * self.target_world + alpha * target_raw
        
        return self.target_world
    
    def _compute_orientation_weight(self) -> float:
        """
        Compute dynamic orientation weight based on distance to target.
        Weight increases as end-effector approaches target.
        
        Returns:
            Orientation weight in [0, orientation_weight_max]
        """
        current_pos, _ = self.ik_controller.get_current_pose(
            self.data, "body", self.ee_id
        )
        distance = np.linalg.norm(self.target_world - current_pos)
        
        # Linear ramp: weight increases as distance decreases
        ramp = np.clip(
            (self.config.orientation_ramp_distance - distance) / 
            self.config.orientation_ramp_distance,
            0.0, 1.0
        )
        
        return self.config.orientation_weight_max * ramp
    
    def _update_mocap_marker(self):
        """Update visual marker to show current target."""
        self.data.mocap_pos[self.mocap_id] = self.target_world
        self.data.mocap_quat[self.mocap_id] = np.array([1, 0, 0, 0])
    
    def _step_physics(self):
        """
        Step physics simulation while keeping arm at commanded position.
        This allows other objects to move while arm is kinematically controlled.
        """
        q_arm = self.data.qpos[:7].copy()
        
        for _ in range(self.config.physics_substeps):
            mujoco.mj_step(self.model, self.data)
            
            # Lock arm joints (kinematic control)
            self.data.qpos[:7] = q_arm
            self.data.qvel[:7] = 0.0
            self.data.qacc[:7] = 0.0
            
            mujoco.mj_forward(self.model, self.data)
    
    def control_step(self) -> float:
        """
        Execute one control step.
        
        Returns:
            Position error after IK solution (meters)
        """
        # Update target position
        target_pos = self._update_target()
        
        # Compute desired orientation
        target_rot = compute_desired_orientation_2(self.model, self.data, self.ee_id)
        
        # Compute dynamic orientation weight
        w_ori = self._compute_orientation_weight()
        
        # Update visual marker
        self._update_mocap_marker()
        
        # Solve IK
        error = self.ik_controller.solve(
            data=self.data,
            handle_kind="body",
            handle_id=self.ee_id,
            target_pos=target_pos,
            target_rot=target_rot,
            w_pos=1.0,
            w_ori=w_ori
        )
        
        # Step physics
        self._step_physics()
        
        return error
    
    def get_visualization(self) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """
        Get current camera view with screw detection visualization.
        
        Returns:
            bgr_image: BGR image for OpenCV display
            screw_center: (u, v) pixel coordinates of detected screw, or None
        """
        rgb, _ = get_rgb_depth(self.renderer, self.data, self.camera_id)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Detect screw in image
        center, _ = detect_screw_center_bgr(bgr)
        
        # Draw detection
        if center:
            u, v = center
            cv2.circle(bgr, (u, v), 6, (0, 255, 0), 2)
            cv2.putText(
                bgr, f"Screw: ({u}, {v})",
                (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1
            )
        
        return bgr, center


class KeyboardHandler:
    """Handle keyboard input for controller."""
    
    INSTRUCTIONS = (
        "Controls: [SPACE/S]=Start | [P]=Pause | [R]=Reset | [ESC]=Quit"
    )
    
    @staticmethod
    def process(key: int, controller: VisualServoController) -> bool:
        """
        Process keyboard input.
        
        Args:
            key: OpenCV key code
            controller: Controller to operate on
        
        Returns:
            True to continue, False to exit
        """
        if key in (ord('s'), ord(' '), ord('S')):
            if not controller.running:
                controller.running = True
                logger.info("▶️  Started")
        
        elif key in (ord('p'), ord('P')):
            if controller.running:
                controller.running = False
                logger.info("⏸️  Paused")
        
        elif key in (ord('r'), ord('R')):
            controller.reset()
            logger.info("🔄 Reset")
        
        elif key == 27:  # ESC
            logger.info("👋 Exit requested")
            return False
        
        return True
    
    @staticmethod
    def draw_status(image: np.ndarray, running: bool):
        """Draw status text on image."""
        status = "RUNNING" if running else "PAUSED"
        color = (0, 255, 0) if running else (0, 255, 255)
        
        cv2.putText(
            image, f"{status} - {KeyboardHandler.INSTRUCTIONS}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2
        )


def main():
    """Main entry point."""
    # Initialize MuJoCo
    model, data, camera_id = build_model()
    
    # Create controller
    controller = VisualServoController(model, data, camera_id)
    
    # Setup OpenCV window
    window_name = "Overhead Camera View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Main loop with passive viewer
    with viewer.launch_passive(model, data) as v:
        logger.info("Starting main loop...")
        
        while v.is_running():
            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            if not KeyboardHandler.process(key, controller):
                break
            
            # Get visualization
            bgr_image, _ = controller.get_visualization()
            KeyboardHandler.draw_status(bgr_image, controller.running)
            cv2.imshow(window_name, bgr_image)
            
            # Execute control if running
            if controller.running:
                error = controller.control_step()
                
                # Optional: log significant errors
                if error > 0.05:
                    logger.warning(f"Large tracking error: {error:.3f}m")
            
            # Sync viewer
            v.sync()
    
    # Cleanup
    cv2.destroyAllWindows()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()