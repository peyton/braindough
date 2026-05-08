#!/usr/bin/env -S just --working-directory . --justfile

[private]
@default:
    just --list

bootstrap:
    mise install --locked
    mise exec -- uv sync --python 3.12.13 --all-packages
    mise exec -- hk install

fmt:
    mise exec -- ruff format packages scripts tests
    mise exec -- ruff check --fix packages scripts tests
    if ls scripts/*.sh >/dev/null 2>&1; then mise exec -- shfmt -w scripts/*.sh; fi

fmt-check:
    mise exec -- ruff format --check packages scripts tests
    if ls scripts/*.sh >/dev/null 2>&1; then mise exec -- shfmt -d scripts/*.sh; fi

lint:
    mise exec -- ruff check packages scripts tests
    mise exec -- actionlint
    env -u GITHUB_TOKEN -u GH_TOKEN -u ZIZMOR_GITHUB_TOKEN mise exec -- zizmor --no-progress .github/workflows

typecheck:
    mise exec -- pyright

test:
    mise exec -- uv run --python 3.12.13 pytest

check: fmt-check lint typecheck test artifact-validate research-validate

ci: bootstrap check run-fake run-fake-optimization run-fake-lesion run-fake-optimizer run-fake-counterfactual

doctor:
    mise exec -- uv run --python 3.12.13 braindough doctor

storage-init:
    mise exec -- uv run --python 3.12.13 braindough storage init

storage-doctor:
    mise exec -- uv run --python 3.12.13 braindough storage doctor

run-fake:
    mise exec -- uv run --python 3.12.13 braindough run experiments/smoke/fake_first_suite.yaml

run-fake-optimization:
    mise exec -- uv run --python 3.12.13 braindough run experiments/smoke/fake_perturbation_optimization.yaml

run-fake-lesion:
    mise exec -- uv run --python 3.12.13 braindough run experiments/smoke/fake_virtual_lesion_lab.yaml

run-fake-optimizer:
    mise exec -- uv run --python 3.12.13 braindough run experiments/smoke/fake_discrete_stimulus_optimizer.yaml

run-fake-counterfactual:
    mise exec -- uv run --python 3.12.13 braindough run experiments/smoke/fake_counterfactual_editing_workbench.yaml

run-tribe:
    mise exec -- uv run --python 3.12.13 --package braindough --extra tribe braindough run experiments/local/tribe_v2_first_suite.yaml

run-tribe-optimization:
    mise exec -- uv run --python 3.12.13 --package braindough --extra tribe braindough run experiments/local/tribe_v2_perturbation_optimization.yaml

run-tribe-lesion:
    mise exec -- uv run --python 3.12.13 --package braindough --extra tribe braindough run experiments/local/tribe_v2_virtual_lesion_lab.yaml

run-tribe-optimizer:
    mise exec -- uv run --python 3.12.13 --package braindough --extra tribe braindough run experiments/local/tribe_v2_discrete_stimulus_optimizer.yaml

run-tribe-counterfactual:
    mise exec -- uv run --python 3.12.13 --package braindough --extra tribe braindough run experiments/local/tribe_v2_counterfactual_editing_workbench.yaml

artifact-validate RUN_DIR='':
    run_dir="{{RUN_DIR}}"; run_dir="${run_dir#RUN_DIR=}"; \
        if [ -n "$run_dir" ]; then \
            mise exec -- uv run --python 3.12.13 braindough validate "$run_dir"; \
        else \
            mise exec -- uv run --python 3.12.13 braindough validate --fixture; \
        fi

research-validate:
    mise exec -- uv run --python 3.12.13 python -m scripts.research_validator

report RUN_DIR:
    run_dir="{{RUN_DIR}}"; run_dir="${run_dir#RUN_DIR=}"; \
        mise exec -- uv run --python 3.12.13 braindough report "$run_dir"

executive-summary RUN_DIRS='' OUTPUT_DIR='':
    run_dirs="{{RUN_DIRS}}"; run_dirs="${run_dirs#RUN_DIRS=}"; \
        output_dir="{{OUTPUT_DIR}}"; output_dir="${output_dir#OUTPUT_DIR=}"; \
        set --; \
        if [ -n "$run_dirs" ]; then \
            old_ifs="$IFS"; IFS='|'; \
            for run_dir in $run_dirs; do set -- "$@" --run-dir "$run_dir"; done; \
            IFS="$old_ifs"; \
        fi; \
        if [ -n "$output_dir" ]; then set -- "$@" --output-dir "$output_dir"; fi; \
        mise exec -- uv run --python 3.12.13 braindough executive-summary "$@"
