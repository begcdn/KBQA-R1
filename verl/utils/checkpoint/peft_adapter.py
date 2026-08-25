import os


def save_peft_adapter(model, full_state_dict, output_dir):
    """Save a gathered FSDP LoRA state without filtering it twice."""
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors import safe_open

    adapter_state_dict = get_peft_model_state_dict(model, state_dict=full_state_dict)
    if not adapter_state_dict:
        sample_keys = list(full_state_dict)[:5]
        raise RuntimeError(
            "FSDP produced no LoRA weights for checkpoint export; "
            f"sample gathered keys: {sample_keys}"
        )

    expected_parameters = sum(tensor.numel() for tensor in adapter_state_dict.values())

    # PeftModel.save_pretrained performs its own adapter-name filtering. Passing
    # adapter_state_dict here would filter the already-normalized keys a second
    # time and can silently create a valid-looking, empty safetensors file.
    model.save_pretrained(
        output_dir,
        state_dict=full_state_dict,
        safe_serialization=True,
    )

    adapter_path = os.path.join(output_dir, "adapter_model.safetensors")
    if not os.path.isfile(adapter_path):
        raise RuntimeError(f"LoRA export did not create {adapter_path}")

    with safe_open(adapter_path, framework="pt", device="cpu") as saved:
        saved_keys = list(saved.keys())
        saved_parameters = sum(saved.get_tensor(key).numel() for key in saved_keys)

    if not saved_keys or saved_parameters != expected_parameters:
        raise RuntimeError(
            "Invalid LoRA checkpoint export: "
            f"expected {len(adapter_state_dict)} tensors/{expected_parameters} parameters, "
            f"saved {len(saved_keys)} tensors/{saved_parameters} parameters"
        )

    return len(saved_keys), saved_parameters
