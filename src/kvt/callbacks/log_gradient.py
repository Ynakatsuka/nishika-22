import pytorch_lightning as pl
import torch


def filter_layers_to_check_gradients(model, max_layers):
    def squueze_layer_names(names):
        result = set()
        for name in sorted(names, key=len):
            skip_flag = bool(sum([1 for r in result if r in name]))
            if not skip_flag:
                result.add(name)
        return result

    layer_names = [n for n, _ in model.named_parameters() if len(n)]

    i = 1
    while True:
        unique_layer_names = set()
        for name in layer_names:
            split_names = name.split(".")
            unique_layer_names.add(
                ".".join(split_names[: max(-i, 1 - len(split_names))])
            )
        i += 1

        # squueze
        unique_layer_names = squueze_layer_names(unique_layer_names)

        if len(unique_layer_names) <= max_layers:
            break

    return unique_layer_names


def aggregate_gradient_norms(model, max_layers=50):
    layers_to_check_gradients = filter_layers_to_check_gradients(
        model, max_layers=max_layers
    )

    gradient_norms = {name: [] for name in layers_to_check_gradients}
    for name, param in model.named_parameters():
        for check_name in layers_to_check_gradients:
            if (
                name.startswith(check_name)
                and param.requires_grad
                and (param.grad is not None)
            ):
                gradient_norms[check_name].append(param.grad.norm(2))
    gradient_norms = {
        k: torch.stack(v).mean() for k, v in gradient_norms.items()
    }
    return gradient_norms


class GradientNormLogger(pl.callbacks.Callback):
    def __init__(self, interval=20):
        self.interval = interval

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.interval == 0:
            gradient_norms = aggregate_gradient_norms(pl_module.model)
            pl_module.log_dict(gradient_norms, on_step=True, sync_dist=True)
