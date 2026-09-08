import gymnasium as _gym

# gymnasium 1.x removed Wrapper.__getattr__; restore delegation to inner env so
# that wrapper attributes (e.g. env.track) are transparently forwarded.
if not hasattr(_gym.Wrapper, "__getattr__"):
    def _wrapper_getattr(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)
    _gym.Wrapper.__getattr__ = _wrapper_getattr

from .action_wrappers import FlattenAction
from .misc_wrappers import FrameSkip
from .cbf_wrappers import CBFSafetyLayer
