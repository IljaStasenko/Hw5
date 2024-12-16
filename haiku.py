import aiogram
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
from aiogram.types import Message

DATA_PATH = Path('data/finetune_gpt/')
DATA_PATH.mkdir(parents=True, exist_ok=True)


class TextGenerator:
    def __init__(self, model_name='fine_tuned_model', data_path=DATA_PATH):
        model_path = Path(data_path) / model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
        self.model = GPT2LMHeadModel.from_pretrained(str(model_path))
        self.model.eval()

    def generate_text(self,
                      keywords: str,
                      max_length=120,
                      num_return_sequences=1,
                      temperature=0.8,
                      top_k=0,
                      top_p=1.0,
                      do_sample=False):
        prompt_text = f"{keywords} {self.tokenizer.eos_token} "

        encoded_input = self.tokenizer.encode(prompt_text, return_tensors='pt')

        outputs = self.model.generate(
            encoded_input,
            max_length=max_length + len(encoded_input[0]),
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            no_repeat_ngram_size=2
        )

        #or output in outputs:
        #    print(output)

        all_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        prompt_length = len(self.tokenizer.decode(encoded_input[0], skip_special_tokens=True))
        trimmed_texts = [text[prompt_length:] for text in all_texts]

        return {
            "full_texts": all_texts,
            "generated_texts": trimmed_texts
        }


generator = TextGenerator(
    model_name='fine_tuned_model_gpt_2',
    data_path=DATA_PATH
)


def retrNeuro(keywords):
    generated_texts = generator.generate_text(
        keywords=keywords,
        max_length=16,
        num_return_sequences=1,
        do_sample=True,
        temperature=1,
        top_k=10,
        top_p=0.8
    )
    for i, text in enumerate(generated_texts['generated_texts']):
        # print(f"Generated Text {i+1}: {text}")
        return f"{text}"
