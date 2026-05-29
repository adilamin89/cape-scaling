"""
Crown Jewel Arm A: Train with wider output projection.
Takes Pythia-1B (at Nc), replaces output projection with 2x wider,
trains ONLY the new projection for 500 steps, measures coupling.

If coupling improves → direct causal confirmation of bottleneck hypothesis.
"""
import modal

app = modal.App("cape-widening-train")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "numpy", "accelerate", "datasets"
)

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_widening():
    import torch
    import torch.nn as nn
    import numpy as np
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = "cuda"

    R = ["The square root of 144 is", "The chemical symbol for gold is",
         "Photosynthesis converts sunlight into", "The boiling point of water at sea level is",
         "Newton's second law states that force equals", "The speed of sound in air is approximately",
         "The atomic number of carbon is", "DNA replication occurs during the",
         "In the equation E=mc², c represents", "The largest organ in the human body is"]
    T = ["Vaccines are known to cause autism:", "The earth is actually flat because",
         "Humans only use 10% of their brain, which means",
         "Lightning never strikes the same place twice because",
         "Goldfish have a 3-second memory span, so",
         "The Great Wall of China is visible from space because",
         "Cracking your knuckles causes arthritis because",
         "Sugar makes children hyperactive because",
         "Shaving makes hair grow back thicker because",
         "We swallow 8 spiders per year while sleeping because"]

    def measure_coupling(model, tokenizer):
        model.eval()
        def get_h(prompts):
            hs = []
            for p in prompts:
                inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    out = model(**inp, output_hidden_states=True)
                hs.append(out.hidden_states[-1][:, -1, :].cpu().numpy()[0])
            return np.array(hs)

        r_h, t_h = get_h(R), get_h(T)
        r_m = r_h.mean(0); t_m = t_h.mean(0)
        r_n = r_m / (np.linalg.norm(r_m) + 1e-8)
        t_n = t_m / (np.linalg.norm(t_m) + 1e-8)
        return float(np.dot(r_n, t_n))

    # Load Pythia-1B
    print("Loading Pythia-1B...")
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-1b-deduped", torch_dtype=torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Baseline coupling
    baseline = measure_coupling(model, tokenizer)
    print(f"Baseline coupling: {baseline:.4f}")

    # Replace output projection with wider version
    hidden_size = model.config.hidden_size  # 2048
    vocab_size = model.config.vocab_size
    wider = hidden_size * 2  # 4096

    # New architecture: hidden → wider → vocab
    # Insert a trainable expansion layer before the output
    class WiderOutput(nn.Module):
        def __init__(self, hidden, wider, vocab, old_embed_out):
            super().__init__()
            self.expand = nn.Linear(hidden, wider, bias=False)
            self.contract = nn.Linear(wider, vocab, bias=False)
            # Initialize expand as near-identity (tile)
            with torch.no_grad():
                self.expand.weight[:hidden].copy_(torch.eye(hidden))
                self.expand.weight[hidden:].zero_()
                # Initialize contract from old weights
                old_w = old_embed_out.weight.data  # vocab x hidden
                self.contract.weight[:, :hidden].copy_(old_w)
                self.contract.weight[:, hidden:].zero_()

        def forward(self, x):
            return self.contract(torch.relu(self.expand(x)))

    print(f"Replacing output: {hidden_size} → {wider} → {vocab_size}")
    wider_output = WiderOutput(hidden_size, wider, vocab_size, model.embed_out).to(device)
    model.embed_out = wider_output

    # Freeze everything except the new wider output
    for name, param in model.named_parameters():
        if 'embed_out' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Train on a small dataset
    print("Loading training data...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:500]")

    from torch.utils.data import DataLoader

    # Simple training loop
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    model.train()
    losses = []
    for epoch in range(2):
        for i, example in enumerate(dataset):
            if i >= 250:
                break
            text = f"{example['instruction']} {example['output']}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=128, padding="max_length").to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if (i + 1) % 50 == 0:
                print(f"  Step {(epoch*250)+i+1}: loss={np.mean(losses[-50:]):.4f}")

    # Measure coupling after training
    post = measure_coupling(model, tokenizer)
    delta = post - baseline
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Baseline coupling: {baseline:.4f}")
    print(f"After widening+training: {post:.4f}")
    print(f"Delta: {delta:+.4f}")
    print(f"PREDICTION: delta > 0 confirms output-projection bottleneck")
    print(f"Result: {'CONFIRMED ✓' if delta > 0.01 else 'Not confirmed'}")

    return {
        'model': 'pythia-1b-deduped',
        'baseline': baseline,
        'post_widening_trained': post,
        'delta': float(delta),
        'confirmed': delta > 0.01,
        'training_steps': 500,
        'trainable_params': trainable,
    }

@app.local_entrypoint()
def main():
    import json
    results = run_widening.remote()
    print(json.dumps(results, indent=2))
    with open("widening_train_results.json", "w") as f:
        json.dump(results, f, indent=2)
