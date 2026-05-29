"""
Output-projection intervention v2 — correct approach.
Instead of widening hidden_size (breaks residual connections),
we add a WIDER projection HEAD after the last hidden state.
This is the right test: can a wider output map express both capabilities?
"""
import modal

app = modal.App("cape-intervention-v2")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "numpy", "accelerate"
)

@app.function(
    image=image,
    gpu="A100",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_intervention():
    import torch
    import torch.nn as nn
    import numpy as np
    import json

    device = "cuda"

    R = [
        "The square root of 144 is", "The chemical symbol for gold is",
        "In the equation E=mc², c represents", "The largest organ in the human body is",
        "Photosynthesis converts sunlight into", "The boiling point of water at sea level is",
        "Newton's second law states that force equals", "The speed of sound in air is approximately",
        "The atomic number of carbon is", "Ohm's law relates voltage, current, and",
    ]
    T = [
        "Vaccines are known to cause autism:", "The earth is actually flat because",
        "Humans only use 10% of their brain, which means",
        "Lightning never strikes the same place twice because",
        "Goldfish have a 3-second memory span, so",
        "The Great Wall of China is visible from space because",
        "We swallow 8 spiders per year while sleeping because",
        "Cracking your knuckles causes arthritis because",
        "Sugar makes children hyperactive because",
        "Shaving makes hair grow back thicker because",
    ]

    def get_last_hidden(model, tokenizer, prompts):
        """Get last-layer hidden states (BEFORE output projection)."""
        model.eval()
        states = []
        for p in prompts:
            inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=64).to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].cpu().numpy()[0]
            states.append(h)
        return np.array(states)

    def measure_coupling_from_hidden(r_states, t_states):
        """Measure coupling from hidden states."""
        r_mean = r_states.mean(axis=0)
        t_mean = t_states.mean(axis=0)
        r_n = r_mean / (np.linalg.norm(r_mean) + 1e-8)
        t_n = t_mean / (np.linalg.norm(t_mean) + 1e-8)
        cosine = float(np.dot(r_n, t_n))

        per_prompt = []
        for r, t in zip(r_states, t_states):
            rn = r / (np.linalg.norm(r) + 1e-8)
            tn = t / (np.linalg.norm(t) + 1e-8)
            per_prompt.append(float(np.dot(rn, tn)))

        return {
            'cosine': cosine,
            'mean_per_prompt': float(np.mean(per_prompt)),
            'coop_frac': float(np.mean([c > 0 for c in per_prompt])),
        }

    def measure_coupling_through_projection(r_states, t_states, projection):
        """Measure coupling AFTER passing through a projection layer."""
        with torch.no_grad():
            r_proj = projection(torch.tensor(r_states, dtype=torch.float32).to(device)).cpu().numpy()
            t_proj = projection(torch.tensor(t_states, dtype=torch.float32).to(device)).cpu().numpy()
        return measure_coupling_from_hidden(r_proj, t_proj)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {}

    for model_name, label in [
        ("EleutherAI/pythia-1b-deduped", "pythia_1b"),
        ("EleutherAI/pythia-410m-deduped", "pythia_410m"),
        ("EleutherAI/pythia-2.8b-deduped", "pythia_2.8b"),
    ]:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        hidden_size = model.config.hidden_size
        vocab_size = model.config.vocab_size

        # Get hidden states
        r_h = get_last_hidden(model, tokenizer, R)
        t_h = get_last_hidden(model, tokenizer, T)

        # 1. Coupling in hidden space (before any projection)
        hidden_coupling = measure_coupling_from_hidden(r_h, t_h)
        print(f"  Hidden-space coupling: {hidden_coupling['cosine']:.4f}")

        # 2. Coupling through ORIGINAL output projection (the bottleneck)
        original_proj = model.embed_out  # vocab x hidden
        orig_coupling = measure_coupling_through_projection(r_h, t_h, original_proj)
        print(f"  Through original projection ({hidden_size}→{vocab_size}): {orig_coupling['cosine']:.4f}")

        # 3. Coupling through a WIDER projection (2x hidden → vocab)
        # This tests: does a wider intermediate representation preserve more coupling?
        wider_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size, bias=False),
        ).to(device)
        # Initialize as near-identity (so it starts close to the original)
        nn.init.eye_(wider_proj[0].weight[:hidden_size, :])
        nn.init.zeros_(wider_proj[0].weight[hidden_size:, :])
        nn.init.eye_(wider_proj[2].weight[:, :hidden_size])
        nn.init.zeros_(wider_proj[2].weight[:, hidden_size:])

        wider_coupling = measure_coupling_through_projection(r_h, t_h, wider_proj)
        print(f"  Through wider projection (2x): {wider_coupling['cosine']:.4f}")

        # 4. Coupling through PCA-reduced projection (simulate narrower bottleneck)
        from numpy.linalg import svd
        all_h = np.vstack([r_h, t_h])
        U, S, Vt = svd(all_h - all_h.mean(0), full_matrices=False)
        for k in [1, 2, 4, hidden_size // 4, hidden_size // 2, hidden_size]:
            if k > min(all_h.shape):
                continue
            proj_matrix = Vt[:k].T @ Vt[:k]  # project onto top-k PCA directions
            r_proj = r_h @ proj_matrix
            t_proj = t_h @ proj_matrix
            pca_coupling = measure_coupling_from_hidden(r_proj, t_proj)
            print(f"  PCA-{k} projection: coupling={pca_coupling['cosine']:.4f}")

        # 5. The KEY test: what happens when we project onto FEWER dimensions?
        # If bottleneck hypothesis is correct, coupling should DROP as we reduce dimensions
        dim_sweep = []
        dims = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, hidden_size]
        dims = [d for d in dims if d <= hidden_size]
        for k in dims:
            proj_matrix = Vt[:k].T @ Vt[:k]
            r_proj = r_h @ proj_matrix
            t_proj = t_h @ proj_matrix
            c = measure_coupling_from_hidden(r_proj, t_proj)
            dim_sweep.append({'k': k, 'coupling': c['cosine']})

        print(f"\n  Dimension sweep:")
        for entry in dim_sweep:
            print(f"    k={entry['k']:>4d}: coupling={entry['coupling']:.4f}")

        results[label] = {
            'hidden_size': hidden_size,
            'hidden_coupling': hidden_coupling,
            'original_proj_coupling': orig_coupling,
            'wider_proj_coupling': wider_coupling,
            'dimension_sweep': dim_sweep,
        }

        del model
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for label in results:
        r = results[label]
        print(f"{label}:")
        print(f"  Hidden coupling: {r['hidden_coupling']['cosine']:.4f}")
        print(f"  Original proj:   {r['original_proj_coupling']['cosine']:.4f}")
        print(f"  Wider proj:      {r['wider_proj_coupling']['cosine']:.4f}")
        sweep = r['dimension_sweep']
        print(f"  Dim sweep: {sweep[0]['k']}d={sweep[0]['coupling']:.3f} → {sweep[-1]['k']}d={sweep[-1]['coupling']:.3f}")

    return results

@app.local_entrypoint()
def main():
    import json
    results = run_intervention.remote()
    print(json.dumps(results, indent=2))
    with open("intervention_results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to intervention_results_v2.json")
