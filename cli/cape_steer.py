"""
CAPE Self-Steering Engine
Activation-level alignment correction at the quarter-depth probe layer.

Works with any HuggingFace transformer model. Automatically detects:
- Number of layers → probe layer = nl // 4
- Hidden dimension → truth direction dimensionality
- Architecture → hook point

Usage:
    from cape_steer import CAPESteerer
    steerer = CAPESteerer("EleutherAI/pythia-410m")
    result = steerer.steer("The earth is flat because")
    print(result.normal_text)
    print(result.steered_text)
    print(result.phase)  # TAX / TRANSITION / BONUS
    print(result.cos_truth)  # cosine with truth direction
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import json


# Default calibration prompts (true vs false statements)
CALIBRATION_TRUE = [
    "Scientific evidence shows vaccines prevent disease",
    "The Earth orbits the Sun approximately every 365 days",
    "Water is composed of hydrogen and oxygen atoms",
    "Evolution by natural selection is supported by fossil evidence",
    "Climate data shows global temperatures have increased",
    "Antibiotics treat bacterial infections but not viral ones",
    "The speed of light in vacuum is approximately 3e8 m/s",
    "DNA contains the genetic instructions for organisms",
]

CALIBRATION_FALSE = [
    "Vaccines are a government plot to control population",
    "The Earth is flat and NASA fakes all space photos",
    "Water is a single element with no internal structure",
    "Evolution has been completely disproven by modern science",
    "Global temperatures have been steadily decreasing for decades",
    "Antibiotics cure all diseases including viral infections",
    "Light travels instantaneously with no speed limit",
    "DNA has no role in heredity or genetic information",
]


@dataclass
class SteerResult:
    """Result of steering a single prompt."""
    prompt: str
    normal_text: str
    steered_text: str
    phase: str          # TAX, TRANSITION, BONUS
    cos_truth: float    # cosine similarity with truth direction
    correction_strength: float
    changed: bool       # whether output differs
    probe_layer: int
    num_layers: int
    model_name: str


class CAPESteerer:
    """
    CAPE Self-Steering: adds a truth-direction vector at the probe layer
    during generation to correct misaligned outputs.

    The probe layer is always at quarter-depth (num_layers // 4),
    where the coupling bottleneck lives (Paper 3A, Section 5).
    """

    def __init__(self, model_name: str, device: str = "auto", dtype=None):
        """
        Args:
            model_name: HuggingFace model name (e.g., "EleutherAI/pythia-410m")
            device: "auto", "cuda", "cpu", or "mps"
            dtype: torch dtype (default: auto-detect)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

        # Detect architecture
        config = getattr(self.model.config, 'text_config', self.model.config)
        self.num_layers = config.num_hidden_layers
        self.hidden_dim = config.hidden_size
        self.probe_layer = self.num_layers // 4

        print(f"  Layers: {self.num_layers}, Hidden: {self.hidden_dim}")
        print(f"  Probe layer: {self.probe_layer} (quarter-depth)")
        print(f"  Device: {self.device}, Dtype: {self.dtype}")

        # Compute truth direction
        self.truth_direction = None
        self._calibrate()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _get_hook_layer(self):
        """Find the correct module to hook for hidden state extraction."""
        model = self.model
        # Try common architecture patterns
        for attr in ['model.layers', 'transformer.h', 'gpt_neox.layers', 'model.decoder.layers']:
            parts = attr.split('.')
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                return obj[self.probe_layer]
            except (AttributeError, IndexError):
                continue
        raise ValueError(f"Cannot find layer modules in {type(model).__name__}. "
                        f"Supported: Llama, GPT-2, GPT-NeoX, Pythia, OPT, Mistral, Gemma, Qwen")

    def _get_hidden_at_probe(self, text: str) -> torch.Tensor:
        """Get the hidden state at the probe layer for the last token."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        hidden_states = []

        def hook_fn(module, input, output):
            # output is typically a tuple; first element is the hidden state
            if isinstance(output, tuple):
                hidden_states.append(output[0][:, -1, :].detach())
            else:
                hidden_states.append(output[:, -1, :].detach())

        layer = self._get_hook_layer()
        handle = layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            self.model(**inputs)

        handle.remove()
        return hidden_states[0] if hidden_states else torch.zeros(1, self.hidden_dim, device=self.device)

    def _calibrate(self, true_prompts=None, false_prompts=None):
        """Compute truth direction from calibration prompts."""
        true_prompts = true_prompts or CALIBRATION_TRUE
        false_prompts = false_prompts or CALIBRATION_FALSE

        print(f"  Calibrating truth direction from {len(true_prompts)} prompt pairs...")

        true_acts = []
        false_acts = []

        for tp in true_prompts:
            true_acts.append(self._get_hidden_at_probe(tp))
        for fp in false_prompts:
            false_acts.append(self._get_hidden_at_probe(fp))

        true_mean = torch.stack(true_acts).mean(dim=0)
        false_mean = torch.stack(false_acts).mean(dim=0)

        direction = true_mean - false_mean
        self.truth_direction = F.normalize(direction, dim=-1)
        print(f"  Truth direction computed (norm: {direction.norm().item():.3f})")

    def classify(self, text: str) -> tuple:
        """
        Classify a prompt's alignment phase.
        Returns (phase, cos_truth, correction_strength).
        """
        hidden = self._get_hidden_at_probe(text)
        hidden_norm = F.normalize(hidden, dim=-1)
        cos = (hidden_norm * self.truth_direction).sum().item()

        if cos < -0.1:
            return "SEVERE_TAX", cos, 3.0
        elif cos < 0.1:
            return "MILD_TAX", cos, 1.5
        elif cos < 0.3:
            return "BALANCED", cos, 0.5
        else:
            return "BONUS", cos, 0.0

    def steer(self, prompt: str, max_new_tokens: int = 50,
              strength: Optional[float] = None) -> SteerResult:
        """
        Generate with and without CAPE steering.

        Args:
            prompt: Input text
            max_new_tokens: How many tokens to generate
            strength: Override auto-detected correction strength

        Returns:
            SteerResult with both outputs and diagnostics
        """
        phase, cos_truth, auto_strength = self.classify(prompt)
        correction_strength = strength if strength is not None else auto_strength

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate WITHOUT steering
        with torch.no_grad():
            normal_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9
            )
        normal_text = self.tokenizer.decode(normal_ids[0], skip_special_tokens=True)

        # Generate WITH steering (add truth direction at probe layer)
        steering_hook = None

        def steering_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
                hs[:, -1, :] += correction_strength * self.truth_direction.squeeze(0)
                return (hs,) + output[1:]
            else:
                output[:, -1, :] += correction_strength * self.truth_direction.squeeze(0)
                return output

        layer = self._get_hook_layer()
        if correction_strength > 0:
            steering_hook = layer.register_forward_hook(steering_fn)

        with torch.no_grad():
            steered_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9
            )
        steered_text = self.tokenizer.decode(steered_ids[0], skip_special_tokens=True)

        if steering_hook:
            steering_hook.remove()

        return SteerResult(
            prompt=prompt,
            normal_text=normal_text,
            steered_text=steered_text,
            phase=phase,
            cos_truth=cos_truth,
            correction_strength=correction_strength,
            changed=normal_text != steered_text,
            probe_layer=self.probe_layer,
            num_layers=self.num_layers,
            model_name=self.model_name,
        )

    def batch_steer(self, prompts: list, **kwargs) -> list:
        """Steer multiple prompts."""
        return [self.steer(p, **kwargs) for p in prompts]

    def to_json(self, results: list) -> str:
        """Export results as JSON."""
        return json.dumps([{
            'prompt': r.prompt,
            'normal': r.normal_text,
            'steered': r.steered_text,
            'phase': r.phase,
            'cos_truth': r.cos_truth,
            'strength': r.correction_strength,
            'changed': r.changed,
            'probe_layer': r.probe_layer,
            'model': r.model_name,
        } for r in results], indent=2)
