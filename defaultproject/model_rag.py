
from utils.etc import hit2docdict
import torch
# Modify
class ModelRAG():
    def __init__(self):
        pass

    def set_model(self, model):
        # 구현한 gpt-small
        self.model = model

    def set_retriever(self, retriever):
        #fine-tune 할때 BM25 사용
        self.retriever = retriever

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def search(self, queries, qids, k=5):
        # Use the retriever to get relevant documents
        list_passages = []
        list_scores = []

        # fill here
        ######
        
        hits = self.retriever.search(queries, k=k)
        qid2docs = hit2docdict(hits)

        for qid in qids:
            docs = qid2docs.get(qid, [])
            passages = [d.get("contents", "") for d in docs]
            scores = [d.get("score", 0.0) for d in docs]
            list_passages.append(passages)
            list_scores.append(scores) 
        
        ######

        return list_passages, list_scores

    # Modify
    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        # Get the relevant documents for each query
        list_passages, list_scores = self.search(queries, qids, k=k)
        
        list_input_text_without_answer = []
        # fill here
        ######
        # -------------------------------------------------
        # Build a retrieval-augmented prompt for each query:
        #   [Context]
        #   <p1>
        #   <p2>
        #   ...
        #
        #   [Question]
        #   <query>
        #
        #   [Answer]
        #
        # This prompt style is model-agnostic and works with
        # GPT-small, custom models, and HF models.
        # -------------------------------------------------
        
        ######
        
        return list_input_text_without_answer

    @torch.no_grad()
    def retrieval_augmented_generate(self, queries, qids,k=5, **kwargs):
        # fill here:
        ######
        
        list_input_text_without_answer = self.make_augmented_inputs_for_generate(
            queries, qids, k=k
        )

        inputs = self.tokenizer(
            list_input_text_without_answer,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        ######

        # # Move batch to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            **kwargs
        )
        
        outputs = outputs[:, inputs['input_ids'].size(1):]

        return outputs
