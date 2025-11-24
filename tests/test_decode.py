def test_gpt2_get_logits(gpt2_model, gpt2_tokenizer, input_text):
    from src.decode import compute_logits

    logits = compute_logits(gpt2_model, gpt2_tokenizer, input_text)
    assert logits.dim() == 2
    assert logits.shape[0] == 1
    assert logits.shape[1] == 50257

