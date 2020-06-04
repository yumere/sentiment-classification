from typing import Tuple

from torch import Tensor
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer, BertConfig


class SentimentClassifier(nn.Module):
    def __init__(self, config: BertConfig):
        super(SentimentClassifier, self).__init__()
        self.bert_model = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
                input_ids: Tensor,
                input_mask: Tensor,
                segment_ids: Tensor,
                labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        outputs = self.bert_model(input_ids=input_ids,
                                  attention_mask=input_mask,
                                  token_type_ids=segment_ids)
        pooled_outputs = outputs[1]
        outputs = self.classifier(self.dropout(pooled_outputs))
        loss = F.cross_entropy(outputs, labels, reduction='none')
        return outputs, loss, F.softmax(outputs, dim=1)


if __name__ == '__main__':
    pass
