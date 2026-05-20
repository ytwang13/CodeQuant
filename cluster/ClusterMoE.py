import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class ClusterMoE(nn.Module):
    def __init__(self, moe_module: nn.Module,
                 calibration_mode: bool,
                 layer_name: str):
        super().__init__()
        self.moe_module = moe_module

        self.layer_name = layer_name

        self.calibration_mode = calibration_mode
        self.calibration_output_hidden = None
        self.calibration_output_router = None


    def set_calibration_mode(self, mode):
        self.calibration_mode = mode


    def _clear_calibration_cache(self):
        if hasattr(self, 'calibration_output_hidden') and self.calibration_output_hidden is not None:
            del self.calibration_output_hidden
            self.calibration_output_hidden = None
        if hasattr(self, 'calibration_output_router') and self.calibration_output_router is not None:
            del self.calibration_output_router
            self.calibration_output_router = None


    def _router_logits_from_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pre-softmax gate logits (HF Qwen3/Mixtral MoE returns hidden states only)."""
        hidden_dim = hidden_states.shape[-1]
        hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)
        gate = getattr(self.moe_module, "gate", None)
        if gate is not None and hasattr(gate, "weight"):
            return F.linear(
                hidden_states_reshaped.to(torch.float32),
                gate.weight.to(torch.float32),
                None,
            )
        raise NotImplementedError(
            f"Cannot extract router logits from {type(self.moe_module).__name__}"
        )


    def _has_cluster_weights(self) -> bool:
        for m in self.moe_module.modules():
            if hasattr(m, "cluster_weight") and isinstance(m.cluster_weight, nn.Parameter):
                return True
        return False


    def _parse_moe_output(self, out, hidden_states_in: torch.Tensor):
        if isinstance(out, tuple):
            hidden_states = out[0]
            router_logits = out[1] if len(out) >= 2 else self._router_logits_from_input(hidden_states_in)
        else:
            hidden_states = out
            router_logits = self._router_logits_from_input(hidden_states_in)
        return hidden_states, router_logits


    def forward(self, *args, **kwargs):
        hidden_states_in = args[0] if args else kwargs["hidden_states"]

        if self.calibration_mode:
            out = self.moe_module(*args, **kwargs)
            hidden_states, router_logits = self._parse_moe_output(out, hidden_states_in)
            self.calibration_output_hidden = hidden_states.detach().to('cpu')
            self.calibration_output_router = router_logits.detach().to('cpu')
            return hidden_states

        if not self._has_cluster_weights():
            # Qwen3 fused experts: no ClusterLinear under mlp; skip MoE fine-tune loss.
            out = self.moe_module(*args, **kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out
            self._clear_calibration_cache()
            return hidden_states.detach()

        hidden_states_in = hidden_states_in.detach()
        with torch.enable_grad():
            out = self.moe_module(hidden_states_in)
            hidden_states, router_logits = self._parse_moe_output(out, hidden_states_in)
            self._loss(hidden_states, router_logits)
        return hidden_states.detach()


    def _loss(self, hidden, router):
        calibrate_output_hidden = self.calibration_output_hidden.to(hidden.device)
        calibrate_output_router = self.calibration_output_router.to(router.device)

        loss_hidden = F.mse_loss(hidden, calibrate_output_hidden)
        loss_router = F.kl_div(F.log_softmax(router, dim=-1),
                               F.log_softmax(calibrate_output_router, dim=-1),
                               reduction='batchmean',
                               log_target=True)

        loss = loss_hidden + loss_router

        for m in self.moe_module.modules():
            if hasattr(m, "cluster_weight") and isinstance(m.cluster_weight, nn.Parameter):
                tqdm.write(f"[INFO] {m.layer_name} gradient computing...")
                if m.cluster_weight.grad is not None:
                    m.cluster_weight.grad.zero_()
        loss.backward()

        del calibrate_output_hidden, calibrate_output_router, loss_hidden, loss_router, loss
        self._clear_calibration_cache()
        torch.cuda.empty_cache()


class ClusterMoE_deepseek(nn.Module):
    def __init__(self, config, moe_module: nn.Module,
                 calibration_mode: bool):
        super().__init__()
        self.moe_module = moe_module

        self.calibration_mode = calibration_mode
        self.calibration_output_hidden = None
        self.calibration_output_router = None

        self.close_form_mode = False

    def set_calibration_mode(self, mode):
        self.calibration_mode = mode

    def forward(self, input, **kwargs):
        input = input.detach()
        with torch.enable_grad():
            hidden_states = self.moe_module(input)

            _, _, h = input.shape
            hidden_states_input = (input).view(-1, h)
            router_logits = F.linear(
                hidden_states_input.type(torch.float32), self.moe_module.gate.weight.type(torch.float32), None
            )

            if not self.close_form_mode:
                if self.calibration_mode:
                    self.calibration_output_hidden = hidden_states.detach()
                    self.calibration_output_router = router_logits.detach()
                    return hidden_states.detach()
                else:
                    self._loss(hidden_states, router_logits)
                    return hidden_states.detach()
            else:
                return hidden_states

    def _loss(self, hidden, router):
        calibrate_output_hidden = self.calibration_output_hidden.to(hidden.device)
        calibrate_output_router = self.calibration_output_router.to(router.device)

        loss_hidden = F.mse_loss(hidden, calibrate_output_hidden)

        loss_router = F.kl_div(F.log_softmax(router, dim=-1),
                               F.log_softmax(calibrate_output_router, dim=-1),
                               reduction='batchmean',
                               log_target=True)
        loss = loss_hidden + loss_router

        for m in self.moe_module.modules():
            if hasattr(m, "cluster_weight") and isinstance(m.cluster_weight, nn.Parameter):
                tqdm.write(f"[INFO] {m.layer_name} gradient computing...")
                if m.cluster_weight.grad is not None:
                    m.cluster_weight.grad.zero_()
        loss.backward()
