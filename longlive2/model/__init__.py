"""Load the selected backend without importing the other model family."""
__all__ = ['DMD', 'CausalDiffusion']
def __getattr__(name):
    if name == 'DMD':
        from .dmd import DMD
        return DMD
    if name == 'CausalDiffusion':
        from .diffusion import CausalDiffusion
        return CausalDiffusion
    raise AttributeError(name)
