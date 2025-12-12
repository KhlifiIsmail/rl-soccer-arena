"""Monkey patch to fix Stable-Baselines3 episode info buffer bug."""

import stable_baselines3.common.on_policy_algorithm as on_policy_module


# Patch the _update_info_buffer method instead
_original_update_info_buffer = on_policy_module.OnPolicyAlgorithm._update_info_buffer


def patched_update_info_buffer(self, infos, dones=None):
    """Patched update method that filters bad entries."""
    # Call original
    _original_update_info_buffer(self, infos, dones)
    
    # Filter out non-dict entries from ep_info_buffer
    if hasattr(self, 'ep_info_buffer') and self.ep_info_buffer:
        self.ep_info_buffer = [
            ep for ep in self.ep_info_buffer 
            if isinstance(ep, dict) and len(ep) > 0
        ]


# Apply the patch
on_policy_module.OnPolicyAlgorithm._update_info_buffer = patched_update_info_buffer

print("[SB3 FIX] Applied monkey patch for episode info buffer bug")