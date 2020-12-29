import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel

class BertModelForCSModelling(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lin = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.lin2 = nn.Linear(config.hidden_size, 1, bias=False)

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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        dropped = self.dropout(sequence_output[:, 0, :].squeeze())
        score = self.lin2(self.layer_norm(self.lin(dropped)))
        return score
