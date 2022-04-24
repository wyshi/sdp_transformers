from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


def is_digit(token):
    return token.strip().isdigit()


def get_public_token_ids(tokenizer, policy_function):
    public_tokens = [
        k
        for k in tokenizer.get_vocab().keys()
        if not (policy_function(k) or policy_function(k.replace("Ġ", "")))
    ]
    tok_id_map = tokenizer.get_vocab()
    public_token_ids = [[tok_id_map[tok]] for tok in public_tokens]
    return public_token_ids


tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("gpt2")
input_context = "My cute dog"
# get tokens of words that should not be generated
public_token_ids = get_public_token_ids(tokenizer, policy_function=is_digit)
# encode input context
input_ids = tokenizer(input_context, return_tensors="pt").input_ids
# generate sequences without allowing bad_words to be generated
outputs = model.generate(
    input_ids=input_ids, max_length=20, do_sample=False, bad_words_ids=public_token_ids
)
print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
