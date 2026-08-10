#!/usr/bin/env bash
set -euo pipefail

# Native HyPER-R1 training entrypoint. The underlying trainer remains the
# released KBQA-R1 GRPO implementation; these settings replace its linear
# rollout state with the executable hypothesis graph.

export hyper_r1_enable=true
export hyper_r1_max_active=${hyper_r1_max_active:-6}
export hyper_r1_max_nodes=${hyper_r1_max_nodes:-24}
export hyper_r1_credit_weight=${hyper_r1_credit_weight:-1.0}
export hyper_r1_budget_cost=${hyper_r1_budget_cost:-0.05}

exec "$(dirname "$0")/train_kbqa_sexpr_generation_grpo.sh" "${1:-hyper-r1}"
