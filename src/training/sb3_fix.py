"""Monkey patch to fix Stable-Baselines3 episode info buffer bug."""

import stable_baselines3.common.on_policy_algorithm as on_policy_module


# Store original learn method ONCE at module level
_original_learn = on_policy_module.OnPolicyAlgorithm.learn


def patched_learn(self, *args, **kwargs):
    """Patched learn method that handles episode info buffer correctly."""
    
    # Store original collect_rollouts
    original_collect = self.collect_rollouts
    
    def safe_collect_rollouts(*collect_args, **collect_kwargs):
        """Wrapper that filters ep_info_buffer."""
        result = original_collect(*collect_args, **collect_kwargs)
        
        # Filter ep_info_buffer to only contain valid dicts
        if hasattr(self, 'ep_info_buffer') and self.ep_info_buffer:
            self.ep_info_buffer = [
                ep for ep in self.ep_info_buffer 
                if isinstance(ep, dict) and len(ep) > 0
            ]
        
        return result
    
    # Temporarily replace collect_rollouts
    self.collect_rollouts = safe_collect_rollouts
    
    try:
        # Call the ORIGINAL learn that we stored at module level
        return _original_learn(self, *args, **kwargs)
    finally:
        # Restore original
        self.collect_rollouts = original_collect


# Apply the patch
on_policy_module.OnPolicyAlgorithm.learn = patched_learn

print("[SB3 FIX] Applied monkey patch for episode info buffer bug")