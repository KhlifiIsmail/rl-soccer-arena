"""Monkey patch to fix Stable-Baselines3 episode info buffer bug.

Patches the collect_rollouts method to clean ep_info_buffer before logging.
"""

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm


# Store original collect_rollouts
_original_collect_rollouts = OnPolicyAlgorithm.collect_rollouts


def patched_collect_rollouts(self, *args, **kwargs):
    """Patched collect_rollouts that cleans ep_info_buffer after collection."""
    
    # Call original method
    result = _original_collect_rollouts(self, *args, **kwargs)
    
    # Clean up ep_info_buffer - remove any non-dict entries
    if hasattr(self, 'ep_info_buffer') and len(self.ep_info_buffer) > 0:
        cleaned_buffer = []
        for item in self.ep_info_buffer:
            if isinstance(item, dict) and all(k in item for k in ['r', 'l', 't']):
                cleaned_buffer.append(item)
        
        # Replace with cleaned buffer
        self.ep_info_buffer.clear()
        self.ep_info_buffer.extend(cleaned_buffer)
    
    return result


# Apply the patch
OnPolicyAlgorithm.collect_rollouts = patched_collect_rollouts


print("[SB3 FIX] Patched collect_rollouts to clean ep_info_buffer")