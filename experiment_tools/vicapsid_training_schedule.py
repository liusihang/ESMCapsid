from __future__ import annotations


def build_step_eval_schedule(
    *,
    eval_steps: int,
    early_stopping_patience: int | None = None,
) -> dict[str, int | str | bool]:
    if eval_steps <= 0:
        raise ValueError(f"eval_steps must be positive, got {eval_steps}")

    schedule: dict[str, int | str | bool] = {
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": int(eval_steps),
        "save_steps": int(eval_steps),
        "load_best_model_at_end": True,
    }
    if early_stopping_patience is not None:
        if early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be positive, got {early_stopping_patience}"
            )
        schedule["early_stopping_patience"] = int(early_stopping_patience)
    return schedule
