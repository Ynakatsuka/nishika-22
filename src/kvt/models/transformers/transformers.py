import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CustomTransformersForSequenceClassification(nn.Module):
    """similar to AutoModelForSequenceClassification,
    but this class can collect last hidden outputs of transformers
    and change pooling layer.
    """

    def __init__(
        self,
        backbone,
        neck,
        head,
        n_collect_hidden_states=0,
        reinit_layers=0,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.n_collect_hidden_states = n_collect_hidden_states

        if reinit_layers > 0:
            logger.info(f"Reinitializing Last {reinit_layers} Layers")
            self._init_weights(self.head)
            for layer in self.backbone.encoder.layer[-reinit_layers:]:
                for module in layer.modules():
                    self._init_weights(module)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Sequential):
            for mdl in module:
                self._init_weights(mdl)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        encoder_output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # keys: last_hidden_state, hidden_states (optimal)

        if self.n_collect_hidden_states > 0:
            hidden_layers = encoder_output["hidden_states"][
                -self.n_collect_hidden_states :
            ]
            hidden_state = torch.cat(
                hidden_layers, dim=-1
            )  # (bs, seq_len, n * embedding_dim)
        else:
            hidden_state = encoder_output[
                "last_hidden_state"
            ]  # (bs, seq_len, embedding_dim)

        pooler_output = self.neck(
            hidden_state, attention_mask
        )  # (bs, embedding_dim)
        output = self.head(pooler_output)  # (bs, num_classes)

        return output
