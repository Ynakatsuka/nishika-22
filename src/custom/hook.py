import kvt
import kvt.hooks


@kvt.HOOKS.register
class SamplePostForwardHook(kvt.hooks.PostForwardHookBase):
    def __init__(self):
        super().__init__()

    def __call__(self, outputs):
        return outputs
