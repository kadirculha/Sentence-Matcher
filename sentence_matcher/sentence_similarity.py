import torch
from transformers import AutoTokenizer, AutoModel


class SentenceMatcher:

    def get_similarity(self, sent1, sent2):
        sentences = [sent1, sent2]

        tokenizer = AutoTokenizer.from_pretrained('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
        model = AutoModel.from_pretrained('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')

        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        return round(torch.nn.functional.cosine_similarity(sentence_embeddings[0].unsqueeze(0),
                                                           sentence_embeddings[1].unsqueeze(0)).item(), 4)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
