import abc

import torch


class PostForwardHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, outputs):
        pass


class DefaultPostForwardHook(PostForwardHookBase):
    def __call__(self, outputs):
        return outputs


class SigmoidPostForwardHook(PostForwardHookBase):
    def __call__(self, outputs):
        return torch.sigmoid(outputs)


class SoftmaxPostForwardHook(PostForwardHookBase):
    def __call__(self, outputs):
        return torch.softmax(outputs, dim=-1)
