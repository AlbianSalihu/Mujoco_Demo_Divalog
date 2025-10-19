"""
Inverse Kinematics controller for Panda robot arm.

Provides 6-DOF (position + orientation) IK solver using damped least squares
with adaptive damping and line search for robustness.
"""
import logging
import mujoco
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IKConfig:
    """Configuration parameters for IK solver."""
    step_size: float = 0.1
    dq_limit: float = 0.01
    damping_base: float = 1e-3
    damping_scale: float = 5.0
    iterations: int = 6
    w_pos: float = 1.0
    w_ori: float = 0.3
    line_search_tries: int = 4
    line_search_decay: float = 0.5


class PandaIKController:
    """Inverse kinematics controller for Panda robot arm."""
    
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.config = IKConfig()
        
        # Cache body/joint IDs
        self._arm_dof_mask = self._build_arm_mask()
        self._controlled_dofs = np.nonzero(self._arm_dof_mask)[0]
        
        # Cache finger IDs for TCP calculation
        self._left_finger_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        self._right_finger_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )
        
        logger.info(f"IK controller initialized with {len(self._controlled_dofs)} DOFs: "
                   f"{self._controlled_dofs.tolist()}")
    
    def _build_arm_mask(self) -> np.ndarray:
        """Build boolean mask for Panda arm DOFs (joints 1-7)."""
        mask = np.zeros(self.model.nv, dtype=bool)
        
        for i in range(1, 8):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            if jid != -1:
                dofadr = self.model.jnt_dofadr[jid]
                mask[dofadr] = True
        
        # Fallback: use all hinge/slide joints if specific joints not found
        if np.count_nonzero(mask) < 5:
            logger.warning("Using fallback: all hinge/slide DOFs")
            mask = np.array([
                self.model.jnt_type[self.model.dof_jntid[d]] in 
                (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
                for d in range(self.model.nv)
            ], dtype=bool)
        
        return mask
    
    def _compute_tcp(self, data: mujoco.MjData, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Tool Center Point (TCP) as midpoint between fingers.
        
        Returns:
            tcp_pos: TCP position in world frame (3,)
            offset: Vector from body COM to TCP in world frame (3,)
        """
        # Try finger midpoint first
        if self._left_finger_id != -1 and self._right_finger_id != -1:
            p_left = data.xpos[self._left_finger_id]
            p_right = data.xpos[self._right_finger_id]
            tcp_pos = 0.5 * (p_left + p_right)
            
            body_com = data.xpos[body_id]
            offset = tcp_pos - body_com
            return tcp_pos, offset
        
        # Fallback: use body COM
        return data.xpos[body_id].copy(), np.zeros(3)
    
    def get_current_pose(self, data: mujoco.MjData, 
                        handle_kind: str, 
                        handle_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current pose of end-effector.
        
        Returns:
            position: (3,) world position
            rotation: (3,3) rotation matrix
        """
        if handle_kind == "site":
            pos = data.site_xpos[handle_id].copy()
            rot = data.site_xmat[handle_id].reshape(3, 3).copy()
        else:  # body
            pos, _ = self._compute_tcp(data, handle_id)
            rot = data.xmat[handle_id].reshape(3, 3).copy()
        
        return pos, rot
    
    @staticmethod
    def _skew_symmetric(v: np.ndarray) -> np.ndarray:
        """Convert vector to skew-symmetric matrix."""
        x, y, z = v
        return np.array([
            [ 0, -z,  y],
            [ z,  0, -x],
            [-y,  x,  0]
        ])
    
    @staticmethod
    def _log_map_SO3(R: np.ndarray) -> np.ndarray:
        """
        Compute logarithmic map of SO(3) rotation matrix.
        Returns axis-angle representation as 3D vector.
        """
        trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(trace)
        
        if theta < 1e-6:
            return np.zeros(3)
        
        omega_hat = (R - R.T) / (2.0 * np.sin(theta))
        return np.array([
            omega_hat[2, 1],
            omega_hat[0, 2],
            omega_hat[1, 0]
        ]) * theta
    
    def _compute_jacobian(self, data: mujoco.MjData,
                         handle_kind: str,
                         handle_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute position and rotation Jacobians at control point.
        
        Returns:
            J_pos: (3, nv) position Jacobian
            J_rot: (3, nv) rotation Jacobian
        """
        J_pos = np.zeros((3, self.model.nv))
        J_rot = np.zeros((3, self.model.nv))
        
        if handle_kind == "site":
            mujoco.mj_jacSite(self.model, data, J_pos, J_rot, handle_id)
        else:  # body
            # Get Jacobian at body COM
            mujoco.mj_jacBody(self.model, data, J_pos, J_rot, handle_id)
            
            # Shift to TCP: J_tcp = J_com - [r]_x * J_rot
            _, offset = self._compute_tcp(data, handle_id)
            if np.linalg.norm(offset) > 1e-6:
                J_pos = J_pos - self._skew_symmetric(offset) @ J_rot
        
        return J_pos, J_rot
    
    def _damped_least_squares(self, J: np.ndarray, 
                             error: np.ndarray, 
                             damping: float) -> np.ndarray:
        """Solve J*dq = e using damped least squares."""
        JT = J.T
        damped_inv = np.linalg.inv(J @ JT + damping * np.eye(J.shape[0]))
        return JT @ damped_inv @ error
    
    def _line_search(self, data: mujoco.MjData,
                    handle_kind: str,
                    handle_id: int,
                    dq_arm: np.ndarray,
                    target_pos: np.ndarray,
                    prev_error: float) -> bool:
        """
        Perform line search to find acceptable step size.
        
        Returns:
            True if an improving step was found and applied
        """
        alpha = self.config.step_size
        qpos_backup = data.qpos.copy()
        
        for _ in range(self.config.line_search_tries):
            # Try this step size
            dq_try = dq_arm * alpha
            dq_full = np.zeros(self.model.nv)
            dq_full[self._arm_dof_mask] = dq_try
            
            data.qpos[:] = qpos_backup
            data.qpos[:7] += dq_full[:7]
            mujoco.mj_forward(self.model, data)
            
            # Compute new error
            current_pos, _ = self.get_current_pose(data, handle_kind, handle_id)
            new_error = np.linalg.norm(target_pos - current_pos)
            
            # Accept if improved
            if new_error < prev_error:
                return True
            
            # Otherwise reduce step size
            alpha *= self.config.line_search_decay
        
        # No improvement found - restore original state
        data.qpos[:] = qpos_backup
        mujoco.mj_forward(self.model, data)
        return False
    
    def solve(self, data: mujoco.MjData,
             handle_kind: str,
             handle_id: int,
             target_pos: np.ndarray,
             target_rot: Optional[np.ndarray] = None,
             w_pos: Optional[float] = None,
             w_ori: Optional[float] = None) -> float:
        """
        Solve IK to reach target pose.
        
        Args:
            data: MuJoCo data structure
            handle_kind: "site" or "body"
            handle_id: MuJoCo body/site ID
            target_pos: Target position (3,) in world frame
            target_rot: Target rotation matrix (3,3), optional
            w_pos: Position weight (overrides config if provided)
            w_ori: Orientation weight (overrides config if provided)
        
        Returns:
            Final position error (scalar)
        """
        w_pos = w_pos if w_pos is not None else self.config.w_pos
        w_ori = w_ori if w_ori is not None else self.config.w_ori
        
        for iteration in range(self.config.iterations):
            # Current pose
            current_pos, current_rot = self.get_current_pose(data, handle_kind, handle_id)
            
            # Compute errors
            e_pos = target_pos - current_pos
            
            if target_rot is not None:
                R_error = target_rot @ current_rot.T
                e_ori = self._log_map_SO3(R_error)
            else:
                e_ori = np.zeros(3)
            
            # Weighted error vector
            error_6d = np.hstack([w_pos * e_pos, w_ori * e_ori])
            
            # Compute Jacobian
            J_pos, J_rot = self._compute_jacobian(data, handle_kind, handle_id)
            J_6d = np.vstack([w_pos * J_pos, w_ori * J_rot])[:, self._arm_dof_mask]
            
            # Check for singularity
            if J_6d.size == 0:
                logger.error("Jacobian is empty - no controlled DOFs")
                return np.linalg.norm(e_pos)
            
            # Adaptive damping
            pos_error_norm = np.linalg.norm(e_pos)
            damping = self.config.damping_base * (
                1.0 + self.config.damping_scale * pos_error_norm
            )
            
            # Solve for joint velocities
            dq_arm = self._damped_least_squares(J_6d, error_6d, damping)
            dq_arm = np.clip(dq_arm, -self.config.dq_limit, self.config.dq_limit)
            
            # Line search for good step
            accepted = self._line_search(
                data, handle_kind, handle_id, dq_arm, target_pos, pos_error_norm
            )
            
            if not accepted:
                logger.debug(f"Line search failed at iteration {iteration}")
        
        # Final error
        final_pos, _ = self.get_current_pose(data, handle_kind, handle_id)
        final_error = np.linalg.norm(target_pos - final_pos)
        
        logger.debug(f"IK converged with error: {final_error:.4f}m")
        return final_error