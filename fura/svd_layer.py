import torch
import torch.nn as nn


VALID_S_MERGED_TO = {
    "frozen",
    "trainable",
    "output",
    "input",
    "split",
    "keep_frozen",
    "keep_trainable",
}


def _raise_if_non_cuda_weight(full_name, weight):
    if weight.is_cuda:
        return
    raise RuntimeError(
        "Linear->SVD conversion requires CUDA weights so decomposition runs on GPU. "
        f"Module '{full_name}' is on device={weight.device}."
    )


def resolve_svd_s_merged_to(train_position, s_merged_to=None):
    if train_position not in {"output", "input", "both"}:
        raise ValueError("train_position must be one of: output, input, both")

    if s_merged_to is None:
        if train_position == "both":
            return "split"
        s_merged_to = "frozen"

    if s_merged_to not in VALID_S_MERGED_TO:
        raise ValueError(
            "s_merged_to must be one of: "
            "frozen, trainable, output, input, split, keep_frozen, keep_trainable"
        )

    if s_merged_to in {"output", "input", "split", "keep_frozen", "keep_trainable"}:
        return s_merged_to

    if s_merged_to == "trainable":
        if train_position == "both":
            return "split"
        return train_position

    # frozen
    if train_position == "both":
        return "split"
    return "input" if train_position == "output" else "output"


class SVDLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(in_features, out_features)

        self.svd_a = nn.Parameter(torch.empty(out_features, self.rank))
        self.svd_b = nn.Parameter(torch.empty(self.rank, in_features))
        self.register_parameter("svd_s", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.svd_a, a=5**0.5)
        nn.init.kaiming_uniform_(self.svd_b, a=5**0.5)
        if self.bias is not None:
            bound = 1 / (self.out_features**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def init_from_linear_weight(
        self,
        weight,
        bias=None,
        s_merged_to=None,
        train_position="output",
    ):
        if not weight.is_cuda:
            raise RuntimeError(
                "SVD initialization requires CUDA weights so decomposition runs on GPU. "
                f"Got device={weight.device}."
            )
        compute_dtype = (
            torch.float32 if weight.dtype in (torch.float16, torch.bfloat16) else weight.dtype
        )
        u, s, vh = torch.linalg.svd(weight.to(dtype=compute_dtype), full_matrices=False)
        s = torch.clamp(s, min=0)

        # MOTIVATION-STUDY HOOK (F1): replace the pretrained left basis u with a
        # random orthogonal matrix of the same shape. V is recomputed so that
        # U (s=I) V^T best-fits W. Exact fit when d_out <= d_in; lossy
        # projection when d_out > d_in (tall, e.g. gate/up_proj). Singular
        # values are flattened to 1 because they no longer correspond to the
        # random basis.
        # Activate by setting env var FURA_SVD_RANDOM_U=1 before launching run_rl.
        import os as _os
        if _os.environ.get("FURA_SVD_RANDOM_U") == "1":
            d_out, r = u.shape  # (d_out, r), r = min(d_out, d_in)
            g = torch.randn(d_out, r, device=u.device, dtype=u.dtype)
            u_rand, _ = torch.linalg.qr(g, mode="reduced")
            # Recompute V so U V^T = U U^T W  (best subspace fit).
            vh = u_rand.T @ weight.to(dtype=compute_dtype)
            u = u_rand
            s = torch.ones_like(s)  # scale is now carried by vh, not s.

        merge_target = resolve_svd_s_merged_to(
            train_position=train_position,
            s_merged_to=s_merged_to,
        )
        if merge_target in {"keep_frozen", "keep_trainable"}:
            a = u
            b = vh
            s_keep = torch.ones(
                self.rank,
                device=weight.device,
                dtype=weight.dtype,
            )
            s_keep[: s.shape[0]] = s.to(device=weight.device, dtype=weight.dtype)
            self.svd_s = nn.Parameter(
                s_keep, requires_grad=(merge_target == "keep_trainable")
            )
        elif merge_target == "split":
            s_sqrt = torch.sqrt(s)
            a = u * s_sqrt.unsqueeze(0)
            b = s_sqrt.unsqueeze(1) * vh
        elif merge_target == "output":
            a = u * s.unsqueeze(0)
            b = vh
        else:
            a = u
            b = s.unsqueeze(1) * vh
        if merge_target not in {"keep_frozen", "keep_trainable"}:
            self.svd_s = None

        self.svd_a.copy_(a.to(device=weight.device, dtype=weight.dtype))
        self.svd_b.copy_(b.to(device=weight.device, dtype=weight.dtype))
        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(device=weight.device, dtype=weight.dtype))

    def materialize_dense_weight(self):
        if self.svd_s is not None:
            return (self.svd_a * self.svd_s.unsqueeze(0)) @ self.svd_b
        return self.svd_a @ self.svd_b

    def forward(self, x):
        return nn.functional.linear(x, self.materialize_dense_weight(), self.bias)


def get_svd_target_module_names(svd_type):
    if svd_type == "all":
        return (
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        )
    if svd_type == "mlp":
        return ("gate_proj", "up_proj", "down_proj")
    if svd_type == "attn":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    raise ValueError("svd_type must be one of: all, mlp, attn")


@torch.no_grad()
def convert_linear_to_svd(
    model,
    skip_names=("lm_head",),
    include_names=None,
    s_merged_to=None,
    train_position="output",
):
    include_name_set = set(include_names) if include_names is not None else None
    modules_to_replace = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = name.split(".")[-1]
        if leaf_name in skip_names:
            continue
        if include_name_set is not None and leaf_name not in include_name_set:
            continue
        modules_to_replace.append((name, module))

    non_cuda_modules = [
        (name, module.weight.device)
        for name, module in modules_to_replace
        if not module.weight.is_cuda
    ]
    if non_cuda_modules:
        preview = ", ".join(
            f"{name}({device})" for name, device in non_cuda_modules[:3]
        )
        if len(non_cuda_modules) > 3:
            preview += f", ... (+{len(non_cuda_modules) - 3} more)"
        raise RuntimeError(
            "Linear->SVD conversion requires all target Linear weights on CUDA. "
            f"Found non-CUDA modules: {preview}"
        )

    print(f"Converting {len(modules_to_replace)} Linear layers to SVD factors")

    for full_name, linear in modules_to_replace:
        _raise_if_non_cuda_weight(full_name, linear.weight)
        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        child_name = path[-1]

        svd_layer = SVDLayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        svd_layer.init_from_linear_weight(
            linear.weight.data,
            linear.bias.data if linear.bias is not None else None,
            s_merged_to=s_merged_to,
            train_position=train_position,
        )
        setattr(parent, child_name, svd_layer)

    print("Finished Linear->SVD conversion")
    return [name for name, _ in modules_to_replace]


def configure_svd_trainability(
    model,
    train_position="output",
    train_bias=True,
    train_embed_lm_head=False,
    train_singular_values=False,
):
    if train_position not in {"output", "input", "both"}:
        raise ValueError("train_position must be one of: output, input, both")

    for p in model.parameters():
        p.requires_grad = False

    num_svd_layers = 0
    tuned_output_cores = 0
    tuned_input_cores = 0
    tuned_biases = 0

    for _, module in model.named_modules():
        if not isinstance(module, SVDLayer):
            continue
        num_svd_layers += 1

        if train_position in {"output", "both"}:
            module.svd_a.requires_grad = True
            tuned_output_cores += 1
        if train_position in {"input", "both"}:
            module.svd_b.requires_grad = True
            tuned_input_cores += 1
        if module.svd_s is not None:
            module.svd_s.requires_grad = bool(train_singular_values)

        if module.bias is not None:
            module.bias.requires_grad = train_bias
            if train_bias:
                tuned_biases += 1

    if train_embed_lm_head:
        for name, module in model.named_modules():
            leaf_name = name.split(".")[-1]
            if leaf_name in ("embed_tokens", "lm_head"):
                for p in module.parameters():
                    p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    return {
        "num_svd_layers": num_svd_layers,
        "tuned_output_cores": tuned_output_cores,
        "tuned_input_cores": tuned_input_cores,
        "tuned_biases": tuned_biases,
        "trainable_param_count": trainable_param_count,
        "total_param_count": total_param_count,
        "trainable_params": trainable_params,
    }
