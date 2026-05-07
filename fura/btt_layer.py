import torch
import torch.nn as nn
from transformers.activations import ACT2FN
import numpy as np
import warnings
import ast
import json
import re


VALID_S_MERGED_TO = {
    "frozen",
    "trainable",
    "output",
    "input",
    "split",
    "keep_frozen",
    "keep_trainable",
}
VALID_BLOCKTT_DECOMP_MODES = {"square", "input_one_block", "output_one_block"}
VALID_BLOCKTT_CONVERT_MODES = {"svd", "qr"}


def normalize_blocktt_convert_mode(mode):
    if not isinstance(mode, str):
        raise ValueError("BlockTT convert_mode must be a string")
    canonical = mode.strip().lower()
    if canonical not in VALID_BLOCKTT_CONVERT_MODES:
        allowed = ", ".join(sorted(VALID_BLOCKTT_CONVERT_MODES))
        raise ValueError(f"BlockTT convert_mode must be one of: {allowed}")
    return canonical
BLOCKTT_DECOMP_MODE_ALIASES = {
    "input": "input_one_block",
    "output": "output_one_block",
    "input_block": "input_one_block",
    "output_block": "output_one_block",
}
BLOCKTT_DECOMP_GROUP_TO_MODULES = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "o": ("o_proj",),
    "mlp_upgate": ("gate_proj", "up_proj"),
    "mlp_down": ("down_proj",),
}
BLOCKTT_DECOMP_GROUP_ALIASES = {
    "mlp_up_gate": "mlp_upgate",
    "upgate": "mlp_upgate",
    "up_gate": "mlp_upgate",
}
_BARE_DICT_KEY_PATTERN = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:')
_BARE_MODE_VALUE_PATTERN = re.compile(
    r'(:\s*)(input_one_block|output_one_block|input|output|input_block|output_block|square)\s*([,}])'
)


def normalize_blocktt_decomp_mode(mode, allow_square=True):
    if not isinstance(mode, str):
        raise ValueError("BlockTT decomp mode must be a string")
    canonical_mode = BLOCKTT_DECOMP_MODE_ALIASES.get(mode.strip(), mode.strip())
    valid_modes = VALID_BLOCKTT_DECOMP_MODES if allow_square else {
        "input_one_block",
        "output_one_block",
    }
    if canonical_mode not in valid_modes:
        allowed = ", ".join(sorted(valid_modes))
        raise ValueError(f"BlockTT decomp mode must be one of: {allowed}")
    return canonical_mode


def _parse_decomp_mode_mapping_literal(raw_value):
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    stripped = raw_value.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None

    parse_attempts = []
    parse_attempts.append(stripped)
    sanitized = _BARE_DICT_KEY_PATTERN.sub(r'\1"\2":', stripped)
    sanitized = _BARE_MODE_VALUE_PATTERN.sub(r'\1"\2"\3', sanitized)
    if sanitized != stripped:
        parse_attempts.append(sanitized)

    last_exc = None
    for candidate in parse_attempts:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
            except (ValueError, SyntaxError, TypeError) as exc:
                last_exc = exc
                continue
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("--decomp-mode dict literal must parse to a dictionary")

    raise ValueError(
        "--decomp-mode dictionary could not be parsed. "
        "Use JSON/Python dict syntax, e.g. "
        '\'{"qkv":"input","o":"output","mlp_upgate":"output","mlp_down":"output"}\''
    ) from last_exc


def resolve_blocktt_decomp_modes(decomp_mode, include_names=None, default_mode="input_one_block"):
    include_name_set = set(include_names) if include_names is not None else None
    default_canonical = normalize_blocktt_decomp_mode(default_mode, allow_square=False)

    parsed_mapping = _parse_decomp_mode_mapping_literal(decomp_mode)
    if parsed_mapping is None:
        scalar_mode = normalize_blocktt_decomp_mode(decomp_mode, allow_square=False)
        if include_name_set is None:
            module_modes = {}
        else:
            module_modes = {module_name: scalar_mode for module_name in include_name_set}
        return scalar_mode, module_modes

    group_modes = {}
    for raw_key, raw_value in parsed_mapping.items():
        if not isinstance(raw_key, str):
            raise ValueError("--decomp-mode dict keys must be strings")
        normalized_key = BLOCKTT_DECOMP_GROUP_ALIASES.get(raw_key.strip(), raw_key.strip())
        if normalized_key not in BLOCKTT_DECOMP_GROUP_TO_MODULES:
            valid_keys = ", ".join(sorted(BLOCKTT_DECOMP_GROUP_TO_MODULES))
            raise ValueError(
                f"Invalid --decomp-mode key '{raw_key}'. Expected one of: {valid_keys}"
            )
        group_modes[normalized_key] = normalize_blocktt_decomp_mode(
            str(raw_value), allow_square=False
        )

    module_modes = {}
    for group_name, module_names in BLOCKTT_DECOMP_GROUP_TO_MODULES.items():
        group_mode = group_modes.get(group_name, default_canonical)
        for module_name in module_names:
            module_modes[module_name] = group_mode

    if include_name_set is not None:
        missing_names = include_name_set - set(module_modes)
        if missing_names:
            raise ValueError(
                f"Cannot resolve decomp modes for modules: {sorted(missing_names)}"
            )
        module_modes = {
            module_name: module_modes[module_name]
            for module_name in include_name_set
        }

    canonical_groups = {
        group_name: group_modes.get(group_name, default_canonical)
        for group_name in BLOCKTT_DECOMP_GROUP_TO_MODULES
    }
    return canonical_groups, module_modes


def _closest_factor_pair(d):
    root = int(d ** 0.5)
    best_a = 1
    best_b = d
    best_diff = best_b - best_a
    for a in range(1, root + 1):
        if d % a == 0:
            b = d // a
            diff = abs(b - a)
            if diff < best_diff:
                best_a, best_b, best_diff = a, b, diff
    return best_a, best_b


BLOCKTT_INPUT_FACTORIZATION_GROUPS = BLOCKTT_DECOMP_GROUP_TO_MODULES
BLOCKTT_INPUT_FACTORIZATION_GROUP_ALIASES = BLOCKTT_DECOMP_GROUP_ALIASES


def _parse_input_factorization_pair(value):
    """Parse a single (n, b) spec from str/tuple/list/None.

    Accepted forms:
      - None: returns None (caller decides default).
      - "head" / "closest": returns the literal string sentinel.
      - "n,b" string (or "(n, b)"): returns (int, int).
      - tuple/list of two ints: returns (int, int).
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"head", "closest"}:
            return s
        s = s.strip("()[] ")
        parts = [p for p in re.split(r"[,\s]+", s) if p]
        if len(parts) != 2:
            raise ValueError(
                f"input_factorization spec must be 'head', 'closest', or 'n,b'; got {value!r}"
            )
        try:
            n, b = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"input_factorization spec must be ints 'n,b'; got {value!r}"
            ) from exc
        return (n, b)
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                f"input_factorization tuple must have length 2; got {value!r}"
            )
        return (int(value[0]), int(value[1]))
    raise ValueError(
        f"input_factorization spec type {type(value).__name__} not supported"
    )


def _parse_input_factorization_arg(value):
    """Parse the user-facing --blocktt_input_factorization argument.

    Returns:
      - None → caller falls back to legacy behavior (factorize_by_head flag).
      - dict[str, spec] → per-group/per-module mapping. Each spec is one of:
            "head", "closest", or (n, b) tuple.
      - "head" / "closest" → scalar sentinel applied to all modules.
      - (n, b) tuple → scalar applied to all modules.

    Accepted user input:
      - None: return None.
      - bare string "head" / "closest" / "n,b": scalar.
      - JSON/Python dict literal mapping group names (qkv, o, mlp_upgate,
        mlp_down) or leaf module names (q_proj, ...) to specs.
    """
    if value is None:
        return None
    parsed_mapping = _parse_decomp_mode_mapping_literal(value) if isinstance(value, str) else None
    if parsed_mapping is None and not isinstance(value, dict):
        # Scalar: a string sentinel or "n,b", or a tuple.
        return _parse_input_factorization_pair(value)
    raw_dict = parsed_mapping if parsed_mapping is not None else value
    out = {}
    for raw_key, raw_value in raw_dict.items():
        if not isinstance(raw_key, str):
            raise ValueError(
                "--blocktt_input_factorization dict keys must be strings"
            )
        key = raw_key.strip()
        normalized_group = BLOCKTT_INPUT_FACTORIZATION_GROUP_ALIASES.get(key, key)
        if normalized_group in BLOCKTT_INPUT_FACTORIZATION_GROUPS:
            for module_name in BLOCKTT_INPUT_FACTORIZATION_GROUPS[normalized_group]:
                out[module_name] = _parse_input_factorization_pair(raw_value)
        else:
            # Treat as a leaf module name (q_proj, down_proj, ...).
            out[key] = _parse_input_factorization_pair(raw_value)
    return out


def _resolve_input_factorization_for_module(
    spec, child_name, in_features, head_factorization,
):
    """Resolve a per-module (n, b) override given the parsed spec.

    Args:
      spec: result of _parse_input_factorization_arg, may be None / dict / str / tuple.
      child_name: leaf module name (e.g. 'q_proj', 'down_proj').
      in_features: incoming dim of the Linear.
      head_factorization: (num_heads, head_dim) computed from model_config, or None.

    Returns:
      None → caller falls back to legacy/factorize_by_head behavior.
      "closest" sentinel → caller uses _closest_factor_pair.
      "head" sentinel → caller uses head_factorization (and falls back to
                        _closest_factor_pair if head_factorization is None,
                        i.e. mlp module without a head structure).
      (n, b) tuple → validated; n*b must equal in_features.
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        if child_name not in spec:
            return None
        chosen = spec[child_name]
    else:
        chosen = spec
    if chosen is None:
        return None
    if isinstance(chosen, str):
        if chosen == "closest":
            return "closest"
        if chosen == "head":
            return "head"
        raise ValueError(
            f"resolved input_factorization spec for '{child_name}' must be "
            f"'head', 'closest', or (n, b); got {chosen!r}"
        )
    n, b = chosen
    if n <= 0 or b <= 0 or n * b != in_features:
        raise ValueError(
            f"input_factorization for '{child_name}' must satisfy n*b={in_features}; "
            f"got (n={n}, b={b})"
        )
    return (n, b)


def _resolve_blocktt_trainable_sides(left_size, right_size, train_position):
    if train_position not in {"small", "large", "both"}:
        raise ValueError("BlockTT train_position must be one of: small, large, both")

    if train_position == "both":
        return True, True
    if train_position == "small":
        # Tie-break to left core for deterministic behavior.
        train_left = left_size <= right_size
        return train_left, not train_left

    # Tie-break to left core for deterministic behavior.
    train_left = left_size >= right_size
    return train_left, not train_left


def resolve_blocktt_s_merged_to(train_position, s_merged_to=None, left_size=None, right_size=None):
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

    if left_size is None or right_size is None:
        raise ValueError("left_size and right_size are required for frozen/trainable aliases")

    train_left, train_right = _resolve_blocktt_trainable_sides(
        left_size=left_size,
        right_size=right_size,
        train_position=train_position,
    )
    if train_left and train_right:
        raise ValueError(
            "BlockTT s_merged_to frozen/trainable is invalid when both cores are trainable. "
            "Use output, input, or split."
        )

    if s_merged_to == "trainable":
        return "output" if train_left else "input"
    return "input" if train_left else "output"


def _raise_if_non_cuda_weight(full_name, weight, allow_non_cuda=False):
    if weight.is_cuda or allow_non_cuda:
        return
    raise RuntimeError(
        "Linear->BTT conversion requires CUDA weights so decomposition runs on GPU. "
        f"Module '{full_name}' is on device={weight.device}."
    )


@torch.no_grad()
def convert_linear_to_btt(
    model,
    btt_rank,
    decomp_mode="square",
    init_mode="default",
    forward_impl="einsum",
    skip_names=("lm_head",),
    include_names=None,
    lr_act=False,
    s_merged_to=None,
    train_position="small",
    factorize_by_head=False,
    model_config=None,
    convert_mode="svd",
    allow_non_cuda=False,
    input_factorization=None,
):
    if btt_rank is None:
        btt_rank = "full"
    parsed_input_factorization = _parse_input_factorization_arg(input_factorization)
    if forward_impl != "einsum":
        warnings.warn(
            "forward_impl is ignored by the canonical BTTLayer implementation.",
            stacklevel=2,
        )
    convert_mode = normalize_blocktt_convert_mode(convert_mode)

    include_name_set = set(include_names) if include_names is not None else None
    if isinstance(decomp_mode, dict):
        normalized_decomp_mode = {
            str(name): normalize_blocktt_decomp_mode(mode, allow_square=False)
            for name, mode in decomp_mode.items()
        }
        default_decomp_mode = None
        decomp_mode_printable = normalized_decomp_mode
    else:
        normalized_decomp_mode = None
        default_decomp_mode = normalize_blocktt_decomp_mode(decomp_mode)
        decomp_mode_printable = default_decomp_mode

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
    if non_cuda_modules and not allow_non_cuda:
        preview = ", ".join(
            f"{name}({device})" for name, device in non_cuda_modules[:3]
        )
        if len(non_cuda_modules) > 3:
            preview += f", ... (+{len(non_cuda_modules) - 3} more)"
        raise RuntimeError(
            "Linear->BTT conversion requires all target Linear weights on CUDA. "
            f"Found non-CUDA modules: {preview}. "
            "Pass allow_non_cuda=True to bypass (used by the layer-streaming "
            "loader: torch.linalg.svd works on CPU, just slower)."
        )
    print(
        f"Converting {len(modules_to_replace)} Linear layers to BTT "
        f"(rank={btt_rank}, decomp_mode={decomp_mode_printable}, init_mode={init_mode}, "
        f"forward_impl={forward_impl}, convert_mode={convert_mode})"
    )

    for full_name, linear in modules_to_replace:
        _raise_if_non_cuda_weight(full_name, linear.weight, allow_non_cuda=allow_non_cuda)
        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        child_name = path[-1]
        layer_decomp_mode = default_decomp_mode
        if normalized_decomp_mode is not None:
            if child_name not in normalized_decomp_mode:
                raise ValueError(
                    f"Missing per-module decomp mode for '{child_name}' in --decomp-mode dict"
                )
            layer_decomp_mode = normalized_decomp_mode[child_name]

        output_factorization = None
        layer_input_factorization = None
        head_input_for_module = None
        if model_config is not None:
            num_heads = getattr(model_config, "num_attention_heads", None)
            num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
            head_dim = getattr(model_config, "head_dim", None)
            if head_dim is None and num_heads is not None:
                hidden_size = getattr(model_config, "hidden_size", None)
                if hidden_size is not None:
                    head_dim = hidden_size // num_heads
            if num_heads is not None and head_dim is not None:
                if factorize_by_head:
                    if child_name == "q_proj":
                        output_factorization = (num_heads, head_dim)
                    elif child_name in ("k_proj", "v_proj"):
                        output_factorization = (num_kv_heads, head_dim)
                    elif child_name == "o_proj":
                        layer_input_factorization = (num_heads, head_dim)
                # For the user-facing "head" sentinel under output_one_block,
                # the input side gets split into (n_heads_in, head_dim) where
                # n_heads_in depends on the module:
                #   q_proj, o_proj : input is hidden_size = num_attention_heads*head_dim
                #   k_proj, v_proj : input is hidden_size = num_attention_heads*head_dim
                # All four therefore use (num_attention_heads, head_dim) when
                # din == num_attention_heads * head_dim. The mlp modules have
                # no head structure on their input side, so head_input remains
                # None and the resolver falls back to closest.
                if child_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    if linear.in_features == num_heads * head_dim:
                        head_input_for_module = (num_heads, head_dim)

        if parsed_input_factorization is not None:
            resolved = _resolve_input_factorization_for_module(
                parsed_input_factorization,
                child_name,
                in_features=linear.in_features,
                head_factorization=head_input_for_module,
            )
            if resolved == "closest":
                layer_input_factorization = None  # let BTTLayer fall back to closest pair
            elif resolved == "head":
                if head_input_for_module is not None:
                    layer_input_factorization = head_input_for_module
                else:
                    layer_input_factorization = None
            elif resolved is not None:
                layer_input_factorization = resolved

        btt_layer = BTTLayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=btt_rank,
            bias=(linear.bias is not None),
            lr_act=lr_act,
            decomp_mode=layer_decomp_mode,
            init_mode=init_mode,
            output_factorization=output_factorization,
            input_factorization=layer_input_factorization,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)

        btt_layer.init_from_linear_weight(
            linear.weight.data,
            linear.bias.data if linear.bias is not None else None,
            s_merged_to=s_merged_to,
            train_position=train_position,
            convert_mode=convert_mode,
            allow_non_cuda=allow_non_cuda,
        )

        setattr(parent, child_name, btt_layer)
    print("Finished Linear->BTT conversion")
    return [name for name, _ in modules_to_replace]


def get_blocktt_target_module_names(blocktt_type):
    if blocktt_type == "all":
        return (
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        )
    if blocktt_type == "mlp":
        return ("gate_proj", "up_proj", "down_proj")
    if blocktt_type == "attn":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    raise ValueError("blocktt_type must be one of: all, mlp, attn")


def configure_blocktt_trainability(
    model,
    train_bias=True,
    train_position="small",
    train_singular_values=False,
):
    if train_position not in {"small", "large", "both"}:
        raise ValueError("BlockTT train_position must be one of: small, large, both")

    for p in model.parameters():
        p.requires_grad = False

    num_btt_layers = 0
    tuned_left_cores = 0
    tuned_right_cores = 0
    tuned_biases = 0

    for _, module in model.named_modules():
        if not isinstance(module, BTTLayer):
            continue
        num_btt_layers += 1

        left_size = module.btt_l.numel()
        right_size = module.btt_r.numel()

        train_left, train_right = _resolve_blocktt_trainable_sides(
            left_size=left_size,
            right_size=right_size,
            train_position=train_position,
        )

        module.btt_l.requires_grad = train_left
        module.btt_r.requires_grad = train_right
        if module.btt_s is not None:
            module.btt_s.requires_grad = bool(train_singular_values)
        tuned_left_cores += int(train_left)
        tuned_right_cores += int(train_right)

        if module.bias is not None:
            module.bias.requires_grad = train_bias
            if train_bias:
                tuned_biases += 1

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    return {
        "num_btt_layers": num_btt_layers,
        "tuned_left_cores": tuned_left_cores,
        "tuned_right_cores": tuned_right_cores,
        "tuned_biases": tuned_biases,
        "trainable_param_count": trainable_param_count,
        "total_param_count": total_param_count,
        "trainable_params": trainable_params,
    }


@torch.no_grad()
def normalize_trainable_blocktt_cores_(model, eps=1e-12):
    try:
        from compress.btt.btt_linear import BTTLinear as _CompressBTTLinear
    except ImportError:
        _CompressBTTLinear = None

    normalized_left = 0
    normalized_right = 0
    for module in model.modules():
        if isinstance(module, BTTLayer):
            matched = True
        elif _CompressBTTLinear is not None and isinstance(module, _CompressBTTLinear):
            matched = True
        else:
            matched = False
        if not matched:
            continue
        if not (hasattr(module, "btt_l") and hasattr(module, "btt_r")):
            continue

        if module.btt_r.requires_grad:
            # Packed shape (n, b, m*r): each fixed (n, m, r) vector over b should be unit norm.
            norms = torch.linalg.vector_norm(module.btt_r, dim=1, keepdim=True).clamp_min(eps)
            module.btt_r.div_(norms)
            normalized_right += 1

        if module.btt_l.requires_grad:
            # Packed shape (m, n*r, a): each fixed (m, n, r) vector over a should be unit norm.
            norms = torch.linalg.vector_norm(module.btt_l, dim=2, keepdim=True).clamp_min(eps)
            module.btt_l.div_(norms)
            normalized_left += 1

    return {
        "normalized_left_cores": normalized_left,
        "normalized_right_cores": normalized_right,
    }



class BTTLayer(nn.Module):
    """
    New BTT layout (2-core) using canonical dimensions:
      b: input block dim
      n: number of input blocks
      m: number of output blocks
      a: output block dim

    Parameter shapes:
      btt_r: (n, b, m * rank)
      btt_l: (m, rank * n, a)
    """

    # Opt-in fused-Step-2 Triton kernel. Default OFF so baseline runs are unchanged.
    use_fused_step2: bool = False

    def __init__(
        self,
        in_features,
        out_features,
        rank,
        bias=True,
        lr_act=True,
        lr_act_type="silu",
        num_cores=2,
        decomp_mode="square",
        init_mode="default",
        output_factorization=None,
        input_factorization=None,
    ):
        super(BTTLayer, self).__init__()

        if num_cores != 2:
            raise NotImplementedError("BTTLayer currently supports only c=2 cores.")

        decomp_mode = normalize_blocktt_decomp_mode(decomp_mode)

        self.in_features = in_features
        self.out_features = out_features
        self.num_cores = num_cores
        self.decomp_mode = decomp_mode
        self.init_mode = init_mode
        self.lr_act = lr_act
        self.lr_act_type = lr_act_type
        self.use_gate_proj = lr_act and lr_act_type == "swiglu"

        if output_factorization is not None:
            out_blocks, out_block_size = output_factorization
            if out_blocks * out_block_size != out_features:
                raise ValueError(
                    f"output_factorization {output_factorization} does not match "
                    f"out_features={out_features}"
                )
        else:
            out_blocks, out_block_size = _closest_factor_pair(out_features)

        if input_factorization is not None:
            in_blocks, in_block_size = input_factorization
            if in_blocks * in_block_size != in_features:
                raise ValueError(
                    f"input_factorization {input_factorization} does not match "
                    f"in_features={in_features}"
                )
        else:
            in_blocks, in_block_size = _closest_factor_pair(in_features)

        if decomp_mode == "square":
            m = out_blocks
            a = out_block_size
            n = in_blocks
            b = in_block_size
        elif decomp_mode == "output_one_block":
            m = 1
            a = out_features
            n = in_blocks
            b = in_block_size
        elif decomp_mode == "input_one_block":
            m = out_blocks
            a = out_block_size
            n = 1
            b = in_features
        else:
            raise ValueError(
                "decomp_mode must be one of: square, output_one_block, input_one_block"
            )

        self.b = b
        self.n = n
        self.m = m
        self.a = a

        if isinstance(rank, str):
            if rank != "full":
                raise ValueError("rank as string must be 'full'")
            resolved_rank = min(self.b, self.a)
        elif isinstance(rank, float):
            if not (0 < rank < 1):
                raise ValueError("rank as float must satisfy 0 < rank < 1")
            target_params = rank * out_features * in_features
            rank_denominator = self.m * self.n * (self.a + self.b)
            approx_rank = target_params / rank_denominator
            low_rank = max(1, int(np.floor(approx_rank)))
            high_rank = max(1, int(np.ceil(approx_rank)))
            low_params = self.m * self.n * low_rank * (self.a + self.b)
            high_params = self.m * self.n * high_rank * (self.a + self.b)
            if abs(low_params - target_params) <= abs(high_params - target_params):
                resolved_rank = low_rank
            else:
                resolved_rank = high_rank
        elif isinstance(rank, int):
            if rank <= 0:
                raise ValueError("rank as integer must be > 0")
            resolved_rank = rank
        else:
            raise TypeError("rank must be an int, a float in (0, 1), or 'full'")

        self.rank = resolved_rank

        if lr_act and (not self.use_gate_proj):
            self.act_fn = ACT2FN[lr_act_type]

        if init_mode == "default":
            target_sdv = (in_features + out_features) ** (-1 / 2)
            self.btt_r = nn.Parameter(
                torch.randn(self.n, self.b, self.m * self.rank)
                / self.rank ** (1 / 4)
                * target_sdv ** (1 / 2)
            )
            if self.use_gate_proj:
                self.btt_g = nn.Parameter(
                    torch.randn(self.n, self.b, self.m * self.rank)
                    / self.rank ** (1 / 4)
                    * target_sdv ** (1 / 2)
                )
            self.btt_l = nn.Parameter(
                torch.randn(self.m, self.rank * self.n, self.a)
                / self.rank ** (1 / 4)
                * target_sdv ** (1 / 2)
            )
        elif init_mode == "mup":
            std_r = np.sqrt(1 / self.b) * min(
                1, np.sqrt((self.m * self.rank) / self.b)
            )
            std_l = np.sqrt(1 / (self.rank * self.n)) * min(
                1, np.sqrt(self.a / (self.rank * self.n))
            )
            self.btt_r = nn.Parameter(
                torch.randn(self.n, self.b, self.m * self.rank) * std_r
            )
            if self.use_gate_proj:
                self.btt_g = nn.Parameter(
                    torch.randn(self.n, self.b, self.m * self.rank) * std_r
                )
            self.btt_l = nn.Parameter(
                torch.randn(self.m, self.rank * self.n, self.a) * std_l
            )
        else:
            raise ValueError("init_mode must be one of: default, mup")

        # Metadata used by optimizer for the canonical new-layout BTT tensors.
        self.btt_r.btt_layout = "new"
        self.btt_r.btt_rank = self.rank
        self.btt_r.btt_b = self.b
        self.btt_r.btt_n = self.n
        self.btt_r.btt_m = self.m
        self.btt_r.btt_a = self.a

        if self.use_gate_proj:
            self.btt_g.btt_layout = "new"
            self.btt_g.btt_rank = self.rank
            self.btt_g.btt_b = self.b
            self.btt_g.btt_n = self.n
            self.btt_g.btt_m = self.m
            self.btt_g.btt_a = self.a

        self.btt_l.btt_layout = "new"
        self.btt_l.btt_rank = self.rank
        self.btt_l.btt_b = self.b
        self.btt_l.btt_n = self.n
        self.btt_l.btt_m = self.m
        self.btt_l.btt_a = self.a
        self.register_parameter("btt_s", None)

        if bias == False:
            self.register_parameter("bias", None)
        else:
            stdv = 1.0 / out_features ** (1 / 2)
            self.bias = torch.nn.Parameter(torch.randn(out_features))
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        return (
            f"mode: {self.decomp_mode}, init: {self.init_mode}, "
            f"blocks: ({self.m}x{self.n}), "
            f"block_size: ({self.a}x{self.b}), "
            f"rank: {self.rank}, "
            f"btt_r: {self.btt_r.shape}, "
            f"btt_g: {self.btt_g.shape if self.use_gate_proj else False}, "
            f"btt_l: {self.btt_l.shape}, "
            f"lr_act: {self.lr_act}, "
            f"lr_act_type: {self.lr_act_type}, "
            f"use_gate_proj: {self.use_gate_proj}, "
            f"bias: {self.bias.shape if self.bias is not None else False}"
        )

    @staticmethod
    @torch.no_grad()
    def _qr_decompose_blocks(blocks, use_rank):
        """Per-block QR/LQ decomposition.

        Picks LQ when a >= b and QR when a < b so the small (k x k) factor is
        always orthogonal and the cores fit the existing BTT layout slots:
            l_used: (B, a, use_rank)
            r_used: (B, use_rank, b)

        For full rank (use_rank == min(a, b)) reconstruction is exact. For
        truncated rank we slice the trailing rows/cols of the triangular
        factor; this is NOT Frobenius-optimal, unlike SVD truncation.
        """
        a = blocks.shape[-2]
        b = blocks.shape[-1]
        if a >= b:
            # LQ: blocks.T = Qp @ Rp, blocks = Rp.T @ Qp.T = L @ Q
            # Qp: (B, b, b), Rp: (B, b, a) -> L = Rp.T: (B, a, b), Q = Qp.T: (B, b, b)
            Qp, Rp = torch.linalg.qr(blocks.transpose(-1, -2), mode="reduced")
            L = Rp.transpose(-1, -2)
            Q = Qp.transpose(-1, -2)
            l_used = L[:, :, :use_rank]
            r_used = Q[:, :use_rank, :]
        else:
            # QR: blocks = Q @ R, Q: (B, a, a), R: (B, a, b)
            Q, R = torch.linalg.qr(blocks, mode="reduced")
            l_used = Q[:, :, :use_rank]
            r_used = R[:, :use_rank, :]
        return l_used.contiguous(), r_used.contiguous()

    @torch.no_grad()
    def init_from_linear_weight(
        self,
        weight,
        bias=None,
        s_merged_to=None,
        train_position="small",
        convert_mode="svd",
        allow_non_cuda=False,
    ):
        convert_mode = normalize_blocktt_convert_mode(convert_mode)
        if not weight.is_cuda and not allow_non_cuda:
            raise RuntimeError(
                "BTT initialization requires CUDA weights so decomposition runs on GPU. "
                f"Got device={weight.device}. "
                "Pass allow_non_cuda=True for layer-streaming loaders that BTT-decompose on CPU."
            )
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Linear weight shape must be {(self.out_features, self.in_features)}, "
                f"got {tuple(weight.shape)}"
            )
        if weight.shape != (self.m * self.a, self.n * self.b):
            raise ValueError(
                f"Linear weight shape {tuple(weight.shape)} not compatible with "
                f"BTT blocks (m={self.m}, n={self.n}, a={self.a}, b={self.b})"
            )

        param_dtype = weight.dtype
        # Dense weight (m*a, n*b) -> block matrix batch (m*n, a, b).
        blocks = weight.reshape(self.m, self.a, self.n, self.b)
        blocks = blocks.permute(0, 2, 1, 3).reshape(self.m * self.n, self.a, self.b)
        decomp_dtype = (
            torch.float32
            if param_dtype in (torch.float16, torch.bfloat16)
            else param_dtype
        )

        max_full_rank = min(self.a, self.b)
        use_rank = min(self.rank, max_full_rank)

        core_l = torch.zeros(
            self.m * self.n,
            self.a,
            self.rank,
            device=weight.device,
            dtype=param_dtype,
        )
        core_r = torch.zeros(
            self.m * self.n,
            self.rank,
            self.b,
            device=weight.device,
            dtype=param_dtype,
        )

        if convert_mode == "qr":
            if s_merged_to in {"keep_frozen", "keep_trainable"}:
                raise ValueError(
                    "convert_mode='qr' is incompatible with s_merged_to in "
                    "{'keep_frozen', 'keep_trainable'}: QR has no singular values to keep."
                )
            if s_merged_to is not None:
                warnings.warn(
                    f"s_merged_to={s_merged_to!r} is ignored under convert_mode='qr' "
                    "(no singular-value scaling exists).",
                    stacklevel=2,
                )
            l_used, r_used = self._qr_decompose_blocks(
                blocks.to(dtype=decomp_dtype), use_rank=use_rank
            )
            core_l[:, :, :use_rank] = l_used.to(dtype=param_dtype)
            core_r[:, :use_rank, :] = r_used.to(dtype=param_dtype)
            self.btt_s = None
        else:
            U, S, Vh = torch.linalg.svd(blocks.to(dtype=decomp_dtype), full_matrices=False)

            merge_target = resolve_blocktt_s_merged_to(
                train_position=train_position,
                s_merged_to=s_merged_to,
                left_size=self.btt_l.numel(),
                right_size=self.btt_r.numel(),
            )
            u_used = U[:, :, :use_rank].to(dtype=param_dtype)
            vh_used = Vh[:, :use_rank, :].to(dtype=param_dtype)
            s_used = torch.clamp(S[:, :use_rank], min=0).to(dtype=param_dtype)

            if merge_target in {"keep_frozen", "keep_trainable"}:
                core_l[:, :, :use_rank] = u_used
                core_r[:, :use_rank, :] = vh_used
                s_keep = torch.zeros(
                    self.m * self.n,
                    self.rank,
                    device=weight.device,
                    dtype=param_dtype,
                )
                s_keep[:, :use_rank] = s_used
                self.btt_s = nn.Parameter(
                    s_keep.reshape(self.m, self.n, self.rank),
                    requires_grad=(merge_target == "keep_trainable"),
                )
            elif merge_target == "split":
                sqrt_s = torch.sqrt(s_used)
                core_l[:, :, :use_rank] = u_used * sqrt_s.unsqueeze(1)
                core_r[:, :use_rank, :] = sqrt_s.unsqueeze(-1) * vh_used
            elif merge_target == "output":
                core_l[:, :, :use_rank] = u_used * s_used.unsqueeze(1)
                core_r[:, :use_rank, :] = vh_used
            else:
                core_l[:, :, :use_rank] = u_used
                core_r[:, :use_rank, :] = s_used.unsqueeze(-1) * vh_used
            if merge_target not in {"keep_frozen", "keep_trainable"}:
                self.btt_s = None

        core_l = core_l.reshape(self.m, self.n, self.a, self.rank)
        core_r = core_r.reshape(self.m, self.n, self.rank, self.b)
        packed_l = core_l.permute(0, 1, 3, 2).reshape(
            self.m, self.rank * self.n, self.a
        )
        packed_r = core_r.permute(1, 3, 0, 2).reshape(
            self.n, self.b, self.m * self.rank
        )

        self.btt_l.data.copy_(packed_l)
        self.btt_r.data.copy_(packed_r)

        if bias is not None:
            if self.bias is None:
                raise ValueError("BTTLayer has no bias parameter but bias tensor was provided")
            self.bias.data.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))
        elif self.bias is not None:
            raise ValueError(
                "BTTLayer has bias parameter but no source bias was provided. "
                "Construct BTTLayer with bias=False for biasless source layers."
            )

    @torch.no_grad()
    def materialize_dense_weight(self):
        if self.lr_act:
            raise ValueError("Dense materialization only supports lr_act=False")
        # Canonical layout:
        # btt_r: (n, b, m*r) -> (m, n, r, b)
        # btt_l: (m, n*r, a) -> (m, n, r, a)
        # W[m, a, n, b] = sum_r btt_l[m, n, r, a] * btt_r[m, n, r, b]
        r = self.btt_r.reshape(self.n, self.b, self.m, self.rank).permute(2, 0, 3, 1)
        l = self.btt_l.reshape(self.m, self.n, self.rank, self.a)
        if self.btt_s is not None:
            l = l * self.btt_s.unsqueeze(-1)
        w_blocks = torch.einsum("mnra,mnrb->mnab", l, r)
        return w_blocks.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"BTTLayer expected last dim {self.in_features}, got {x.shape[-1]}"
            )

        orig_shape = x.shape
        x = x.reshape(-1, self.n, self.b)  # (B, n, b)
        batch_n = x.shape[0]
        x_t = x.transpose(0, 1).contiguous()

        # Step 1: (n, B, b) @ (n, b, m*r) -> (n, B, m*r)
        inner_up = torch.bmm(x_t, self.btt_r)
        inner_up = inner_up.reshape(self.n, batch_n, self.m, self.rank)
        inner_up = inner_up.permute(2, 1, 0, 3).contiguous()  # (m, B, n, r)

        if self.use_gate_proj:
            inner_gate = torch.bmm(x_t, self.btt_g)
            inner_gate = inner_gate.reshape(self.n, batch_n, self.m, self.rank)
            inner_gate = inner_gate.permute(2, 1, 0, 3).contiguous()
            inner = torch.nn.functional.silu(inner_gate) * inner_up
        else:
            inner = inner_up
            if hasattr(self, "act_fn"):
                inner = self.act_fn(inner)

        # Step 2: (m, B, n*r) @ (m, n*r, a) -> (m, B, a)
        if BTTLayer.use_fused_step2 and inner.is_cuda:
            from fura_kernels import step2_s_scaled_bmm
            out = step2_s_scaled_bmm(
                inner.reshape(self.m, batch_n, self.rank * self.n),
                self.btt_l,
                self.btt_s,
            )
        else:
            btt_l = self.btt_l
            if self.btt_s is not None:
                btt_l = (
                    self.btt_l.reshape(self.m, self.n, self.rank, self.a)
                    * self.btt_s.unsqueeze(-1)
                ).reshape(self.m, self.rank * self.n, self.a)
            out = torch.bmm(
                inner.reshape(self.m, batch_n, self.rank * self.n),
                btt_l,
            )
        out = out.permute(1, 0, 2).contiguous().reshape(
            *orig_shape[:-1], self.out_features
        )

        if self.bias is not None:
            out += self.bias

        return out


try:
    import bitsandbytes as _bnb
    _HAS_BNB = True
except ImportError:
    _bnb = None
    _HAS_BNB = False


class QBTTLayer(BTTLayer):
    """BTTLayer with the frozen core stored as NF4 via bitsandbytes.

    The frozen side is determined by which BTTLayer core has requires_grad=False
    after configure_blocktt_trainability. The trainable core and btt_s remain as
    regular bf16 nn.Parameters.

    Attributes:
      _qfura_layout: "flat" or "per_core_block".
      _qfura_frozen_side: "btt_l" or "btt_r".
      _qfura_frozen_shape: tuple, original 3D shape of the frozen core.
      _qfura_frozen_dtype: torch.dtype, dtype pre-quantization.
      _qfura_compute_dtype: torch.dtype, dtype to use when dequanting back for forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._qfura_layout = None
        self._qfura_frozen_side = None
        self._qfura_frozen_shape = None
        self._qfura_frozen_dtype = None
        self._qfura_compute_dtype = None

    def _dequantize_frozen_core(self):
        """Dequant the frozen NF4 core back to its original 3D bf16 layout."""
        assert self._qfura_compute_dtype is not None, (
            "_dequantize_frozen_core called on a QBTTLayer without "
            "_qfura_compute_dtype set; use quantize_frozen_core_() to initialize."
        )
        if self._qfura_layout == "flat":
            # _qfura_frozen_flat is a Params4bit of shape (numel, 1).
            dequanted = _bnb.functional.dequantize_4bit(
                self._qfura_frozen_flat.data,
                quant_state=self._qfura_frozen_flat.quant_state,
            )
            return dequanted.reshape(self._qfura_frozen_shape).to(
                self._qfura_compute_dtype
            )
        elif self._qfura_layout == "per_core_block":
            blocks = []
            for params_4bit in self._qfura_frozen_blocks:
                deq = _bnb.functional.dequantize_4bit(
                    params_4bit.data, quant_state=params_4bit.quant_state
                )
                blocks.append(deq)
            stacked = torch.stack(blocks, dim=0)
            return stacked.reshape(self._qfura_frozen_shape).to(
                self._qfura_compute_dtype
            )
        else:
            raise RuntimeError(
                f"QBTTLayer has invalid _qfura_layout: {self._qfura_layout}"
            )

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"QBTTLayer expected last dim {self.in_features}, got {x.shape[-1]}"
            )

        # Dequant the frozen core to its original 3D bf16 layout.
        frozen_dequanted = self._dequantize_frozen_core()
        if self._qfura_frozen_side == "btt_l":
            btt_l = frozen_dequanted
            btt_r = self.btt_r
        else:
            btt_l = self.btt_l
            btt_r = frozen_dequanted

        orig_shape = x.shape
        x = x.reshape(-1, self.n, self.b)
        batch_n = x.shape[0]
        x_t = x.transpose(0, 1).contiguous()

        # Step 1: (n, B, b) @ (n, b, m*r) -> (n, B, m*r)
        inner_up = torch.bmm(x_t, btt_r)
        inner_up = inner_up.reshape(self.n, batch_n, self.m, self.rank)
        inner_up = inner_up.permute(2, 1, 0, 3).contiguous()

        if self.use_gate_proj:
            inner_gate = torch.bmm(x_t, self.btt_g)
            inner_gate = inner_gate.reshape(self.n, batch_n, self.m, self.rank)
            inner_gate = inner_gate.permute(2, 1, 0, 3).contiguous()
            inner = torch.nn.functional.silu(inner_gate) * inner_up
        else:
            inner = inner_up
            if hasattr(self, "act_fn"):
                inner = self.act_fn(inner)

        # Step 2: (m, B, n*r) @ (m, n*r, a) -> (m, B, a)
        if BTTLayer.use_fused_step2 and inner.is_cuda:
            from fura_kernels import step2_s_scaled_bmm
            out = step2_s_scaled_bmm(
                inner.reshape(self.m, batch_n, self.rank * self.n),
                btt_l,
                self.btt_s,
            )
        else:
            if self.btt_s is not None:
                btt_l = (
                    btt_l.reshape(self.m, self.n, self.rank, self.a)
                    * self.btt_s.unsqueeze(-1)
                ).reshape(self.m, self.rank * self.n, self.a)
            out = torch.bmm(
                inner.reshape(self.m, batch_n, self.rank * self.n),
                btt_l,
            )
        out = out.permute(1, 0, 2).contiguous().reshape(
            *orig_shape[:-1], self.out_features
        )

        if self.bias is not None:
            out += self.bias

        return out

    @torch.no_grad()
    def materialize_dense_weight(self):
        """Dequant the frozen core and materialize the dense bf16 weight.

        Used at checkpoint save time: the saved HF-format weight is the dequanted
        BTT factorization, which introduces a one-shot quantization round-trip
        error (documented in the design spec).
        """
        if self.lr_act:
            raise ValueError("Dense materialization only supports lr_act=False")
        frozen_dequanted = self._dequantize_frozen_core()
        if self._qfura_frozen_side == "btt_l":
            btt_l = frozen_dequanted
            btt_r = self.btt_r
        else:
            btt_l = self.btt_l
            btt_r = frozen_dequanted
        r = btt_r.reshape(self.n, self.b, self.m, self.rank).permute(2, 0, 3, 1)
        l = btt_l.reshape(self.m, self.n, self.rank, self.a)
        if self.btt_s is not None:
            l = l * self.btt_s.unsqueeze(-1)
        w_blocks = torch.einsum("mnra,mnrb->mnab", l, r)
        return w_blocks.permute(0, 2, 1, 3).reshape(
            self.out_features, self.in_features
        )


def _pick_frozen_side(btt_layer):
    """Return 'btt_l' or 'btt_r' depending on which side is frozen after
    configure_blocktt_trainability."""
    l_trainable = btt_layer.btt_l.requires_grad
    r_trainable = btt_layer.btt_r.requires_grad
    if l_trainable and r_trainable:
        raise ValueError(
            "quantize_frozen_core_ requires exactly one frozen BTT core. "
            "Both btt_l and btt_r have requires_grad=True; this is train_position='both'."
        )
    if not l_trainable and not r_trainable:
        raise ValueError(
            "quantize_frozen_core_ requires exactly one frozen BTT core. "
            "Neither btt_l nor btt_r has requires_grad=True; trainability was not configured."
        )
    return "btt_r" if l_trainable else "btt_l"


def quantize_frozen_core_(
    btt_layer,
    layout,
    compute_dtype=torch.bfloat16,
    double_quant=True,
    quant_type="nf4",
):
    """Mutate `btt_layer` in place: replace its frozen BTT core with an NF4 blob.

    Args:
        btt_layer: A BTTLayer (not yet a QBTTLayer) with trainability already
            configured via configure_blocktt_trainability so exactly one of
            btt_l / btt_r has requires_grad=False.
        layout: "flat" — pack the entire frozen core into one Params4bit;
                "per_core_block" — pack each outermost-dim block into its own
                Params4bit (one per m for btt_l, one per n for btt_r).
        compute_dtype: dtype the dequanted tensor is cast to inside
            _dequantize_frozen_core. The forward pass and downstream bmm/einsum
            run in this dtype. Default torch.bfloat16.
        double_quant: pass-through to bnb's compress_statistics. Quantizes the
            quantization absmax values for ~0.4 extra bits saved per param.
        quant_type: "nf4" (default) or "fp4". Forwarded to Params4bit.

    Returns:
        The same Python object, reclassed to QBTTLayer via __class__
        assignment. The frozen-side btt_l / btt_r attribute is removed; the
        Params4bit blob(s) are registered as parameter(s) under
        _qfura_frozen_flat (flat layout) or _qfura_frozen_block_{i} (per-block).
    """
    if not _HAS_BNB:
        raise ImportError("bitsandbytes is not installed; required for qfura")
    if layout not in {"flat", "per_core_block"}:
        raise ValueError("layout must be 'flat' or 'per_core_block'")
    if not isinstance(btt_layer, BTTLayer):
        raise TypeError(f"expected BTTLayer, got {type(btt_layer).__name__}")
    if isinstance(btt_layer, QBTTLayer):
        raise ValueError(
            "quantize_frozen_core_ called on a QBTTLayer that is already quantized."
        )

    frozen_side = _pick_frozen_side(btt_layer)
    frozen_param = getattr(btt_layer, frozen_side)
    frozen_shape = tuple(frozen_param.shape)
    frozen_dtype = frozen_param.dtype

    # Reclass to QBTTLayer without re-running __init__.
    btt_layer.__class__ = QBTTLayer
    btt_layer._qfura_layout = layout
    btt_layer._qfura_frozen_side = frozen_side
    btt_layer._qfura_frozen_shape = frozen_shape
    btt_layer._qfura_frozen_dtype = frozen_dtype
    btt_layer._qfura_compute_dtype = compute_dtype

    if layout == "flat":
        flat = frozen_param.detach().reshape(-1, 1).contiguous()
        p4 = _bnb.nn.Params4bit(
            flat,
            requires_grad=False,
            quant_type=quant_type,
            compress_statistics=double_quant,
            quant_storage=torch.uint8,
        )
        # Params4bit triggers NF4 quantization on .to(device). Must land on CUDA.
        p4 = p4.to(device=frozen_param.device)
        btt_layer.register_parameter("_qfura_frozen_flat", p4)
    else:  # per_core_block
        # btt_l shape (m, rank*n, a): one block per m axis.
        # btt_r shape (n, b, m*rank): one block per n axis.
        block_list = []
        outer = frozen_shape[0]
        for i in range(outer):
            block = frozen_param[i].detach().contiguous()
            p4 = _bnb.nn.Params4bit(
                block,
                requires_grad=False,
                quant_type=quant_type,
                compress_statistics=double_quant,
                quant_storage=torch.uint8,
            )
            p4 = p4.to(device=frozen_param.device)
            btt_layer.register_parameter(f"_qfura_frozen_block_{i}", p4)
            block_list.append(p4)
        # Keep convenience list for _dequantize_frozen_core iteration.
        # Use object.__setattr__ to bypass nn.Module's __setattr__ magic on lists.
        object.__setattr__(btt_layer, "_qfura_frozen_blocks", block_list)

    # Remove the frozen core from the module's parameter list.
    delattr(btt_layer, frozen_side)

    return btt_layer


def convert_btt_to_qbtt_(model, layout):
    """Walk `model` and replace every BTTLayer (with trainability configured)
    with a QBTTLayer. Returns stats dict."""
    num_converted = 0
    bytes_saved = 0
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, BTTLayer):
            continue
        if isinstance(module, QBTTLayer):
            continue  # already converted
        # Only convert if exactly one core is frozen.
        l_train = module.btt_l.requires_grad
        r_train = module.btt_r.requires_grad
        if l_train == r_train:
            continue
        frozen_param = module.btt_r if l_train else module.btt_l
        bf16_bytes = frozen_param.numel() * 2  # bf16 is 2 bytes/elem
        nf4_bytes = frozen_param.numel() // 2  # nf4 is 0.5 bytes/elem
        bytes_saved += bf16_bytes - nf4_bytes
        quantize_frozen_core_(module, layout=layout)
        num_converted += 1
        names.append(name)
    return {
        "num_converted": num_converted,
        "bytes_saved": bytes_saved,
        "layout": layout,
        "names": names,
    }


@torch.no_grad()
def convert_and_quantize_linear_to_qbtt_streaming(
    model,
    btt_rank,
    decomp_mode,
    train_position,
    s_merged_to,
    quant_layout,
    target_modules,
    cuda_device,
    skip_names=("lm_head",),
    factorize_by_head=False,
    convert_mode="svd",
    init_mode="default",
    progress_every=10,
    input_factorization=None,
):
    """Layer-streaming BTT + NF4 quantization, designed to fit Llama-3-70B
    + qfura on a single H100.

    Walks every targeted nn.Linear in `model`, and for each one:
      1. Moves the source bf16 weight to `cuda_device` (transient).
      2. Builds a BTTLayer on `cuda_device`, init via SVD/QR.
      3. NF4-quantizes the frozen core via `quantize_frozen_core_`.
      4. Replaces the original linear with the now-QBTTLayer (whose frozen
         core is already on GPU as a Params4bit blob, ~25% the size).
      5. Frees the original bf16 weight on CPU.

    The non-target modules (embed, lm_head, layernorms) stay where they are.
    A separate post-step moves them to GPU in bf16.

    Peak GPU memory during conversion is bounded by **one bf16 layer**
    (down_proj is largest at ~470 MB for Llama-3-70B), so this works on a
    94 GB H100 even though the full bf16 model is ~140 GB.

    Args:
      model: full model on CPU in bf16. Will be mutated in place.
      btt_rank, decomp_mode, train_position, s_merged_to, quant_layout,
      factorize_by_head, convert_mode, init_mode: same semantics as
        `convert_linear_to_btt` + `convert_btt_to_qbtt_`.
      target_modules: iterable of leaf names to convert (e.g. ["q_proj", ...]).
      cuda_device: torch.device (e.g. torch.device("cuda:0")).
      skip_names: leaf names to NOT convert (default: ("lm_head",)).
      progress_every: print progress every N layers converted.

    Returns:
      dict with keys: num_converted, bytes_saved, names, decomp_mode_used.
    """
    if isinstance(decomp_mode, dict):
        normalized_decomp_mode = {
            str(name): normalize_blocktt_decomp_mode(mode, allow_square=False)
            for name, mode in decomp_mode.items()
        }
        default_decomp_mode = None
    else:
        normalized_decomp_mode = None
        default_decomp_mode = normalize_blocktt_decomp_mode(decomp_mode)

    target_set = set(target_modules)
    convert_mode = normalize_blocktt_convert_mode(convert_mode)
    parsed_input_factorization = _parse_input_factorization_arg(input_factorization)

    # Snapshot the list of modules first (we mutate the tree as we go).
    modules_to_replace = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in skip_names:
            continue
        if leaf not in target_set:
            continue
        modules_to_replace.append((name, module))

    print(
        f"[qfura streaming] converting {len(modules_to_replace)} Linear "
        f"layers via {cuda_device} (rank={btt_rank}, decomp={default_decomp_mode}, "
        f"layout={quant_layout})"
    )

    bytes_saved_total = 0
    names = []
    for idx, (full_name, linear) in enumerate(modules_to_replace):
        leaf = full_name.split(".")[-1]
        layer_decomp_mode = default_decomp_mode
        if normalized_decomp_mode is not None:
            if leaf not in normalized_decomp_mode:
                raise ValueError(f"Missing per-module decomp mode for '{leaf}'")
            layer_decomp_mode = normalized_decomp_mode[leaf]

        # Optional per-head factorization (matches non-streaming path).
        output_factorization = None
        layer_input_factorization = None
        head_input_for_module = None
        cfg = getattr(model, "config", None)
        if cfg is not None:
            num_heads = getattr(cfg, "num_attention_heads", None)
            num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
            head_dim = getattr(cfg, "head_dim", None)
            if head_dim is None and num_heads is not None:
                hidden_size = getattr(cfg, "hidden_size", None)
                if hidden_size is not None:
                    head_dim = hidden_size // num_heads
            if num_heads is not None and head_dim is not None:
                if factorize_by_head:
                    if leaf == "q_proj":
                        output_factorization = (num_heads, head_dim)
                    elif leaf in ("k_proj", "v_proj"):
                        output_factorization = (num_kv_heads, head_dim)
                    elif leaf == "o_proj":
                        layer_input_factorization = (num_heads, head_dim)
                if leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    if linear.in_features == num_heads * head_dim:
                        head_input_for_module = (num_heads, head_dim)

        if parsed_input_factorization is not None:
            resolved = _resolve_input_factorization_for_module(
                parsed_input_factorization,
                leaf,
                in_features=linear.in_features,
                head_factorization=head_input_for_module,
            )
            if resolved == "closest":
                layer_input_factorization = None
            elif resolved == "head":
                if head_input_for_module is not None:
                    layer_input_factorization = head_input_for_module
                else:
                    layer_input_factorization = None
            elif resolved is not None:
                layer_input_factorization = resolved

        # 1. Stage source weight on CUDA.
        src_weight = linear.weight.data.to(device=cuda_device, non_blocking=False)
        src_bias = (
            linear.bias.data.to(device=cuda_device) if linear.bias is not None else None
        )

        # 2. Build BTTLayer on CUDA, init from staged weight.
        btt_layer = BTTLayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=btt_rank,
            bias=(linear.bias is not None),
            lr_act=False,
            decomp_mode=layer_decomp_mode,
            init_mode=init_mode,
            output_factorization=output_factorization,
            input_factorization=layer_input_factorization,
        ).to(device=cuda_device, dtype=src_weight.dtype)
        btt_layer.init_from_linear_weight(
            src_weight,
            src_bias,
            s_merged_to=s_merged_to,
            train_position=train_position,
            convert_mode=convert_mode,
        )

        # 3. Configure trainability for this single layer (mirrors
        # configure_blocktt_trainability's per-layer logic).
        left_size = btt_layer.btt_l.numel()
        right_size = btt_layer.btt_r.numel()
        train_left, train_right = _resolve_blocktt_trainable_sides(
            left_size, right_size, train_position
        )
        btt_layer.btt_l.requires_grad = train_left
        btt_layer.btt_r.requires_grad = train_right
        if hasattr(btt_layer, "btt_s") and btt_layer.btt_s is not None:
            btt_layer.btt_s.requires_grad = (s_merged_to == "keep_trainable")
        if btt_layer.bias is not None:
            btt_layer.bias.requires_grad = True  # bias rides with the trainable side

        # 4. NF4-quantize the frozen core in place.
        bf16_bytes_pre = (
            btt_layer.btt_r.numel() * 2 if not train_right else btt_layer.btt_l.numel() * 2
        )
        quantize_frozen_core_(btt_layer, layout=quant_layout)
        bytes_saved_total += bf16_bytes_pre - bf16_bytes_pre // 4  # 4-bit = 1/4 of bf16

        # 5. Splice into model tree, freeing the original Linear (and its CPU bf16 weight).
        path = full_name.split(".")
        parent = model
        for key in path[:-1]:
            parent = getattr(parent, key)
        setattr(parent, path[-1], btt_layer)

        # Free the staged source weight; the BTT layer now holds its own NF4 copy.
        del src_weight, src_bias, linear
        names.append(full_name)

        if (idx + 1) % progress_every == 0 or (idx + 1) == len(modules_to_replace):
            torch.cuda.synchronize()
            mem_alloc = torch.cuda.memory_allocated(cuda_device) / 1e9
            mem_peak = torch.cuda.max_memory_allocated(cuda_device) / 1e9
            print(
                f"[qfura streaming] {idx+1}/{len(modules_to_replace)} done "
                f"(GPU alloc={mem_alloc:.1f}GB, peak={mem_peak:.1f}GB)"
            )
            torch.cuda.reset_peak_memory_stats(cuda_device)

    return {
        "num_converted": len(modules_to_replace),
        "bytes_saved": bytes_saved_total,
        "names": names,
        "decomp_mode_used": default_decomp_mode if default_decomp_mode is not None else "(per-module dict)",
    }


import os as _os
if _os.environ.get("FURA_FUSED_STEP2") == "1":
    BTTLayer.use_fused_step2 = True
