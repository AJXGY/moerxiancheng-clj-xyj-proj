#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Ensure runtime libs are visible for torch/torch_musa imports.
EXTRA_LD_PATHS=()
for candidate in \
	"/home/o_mabin/.local/gfortran/usr/lib/x86_64-linux-gnu" \
	"/home/o_mabin/.local/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread" \
	"/home/o_mabin/.local/musa_toolkits/musa_toolkits_4.2.0/lib" \
	"/home/o_mabin/.local/mudnn/mudnn/lib" \
	"/usr/local/musa/lib"
do
	if [[ -d "${candidate}" ]]; then
		EXTRA_LD_PATHS+=("${candidate}")
	fi
done

if [[ ${#EXTRA_LD_PATHS[@]} -gt 0 ]]; then
	EXTRA_JOINED="$(IFS=:; echo "${EXTRA_LD_PATHS[*]}")"
	export LD_LIBRARY_PATH="${EXTRA_JOINED}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

SINGLE_ONLY="false"
while [[ $# -gt 0 ]]; do
	case "$1" in
		--single-only)
			SINGLE_ONLY="true"
			shift
			;;
		*)
			echo "Unknown arg: $1" >&2
			exit 1
			;;
	esac
done

BENCH_ARGS=()
if [[ "${SINGLE_ONLY}" == "true" ]]; then
	BENCH_ARGS+=(--single-only)
fi

python3 benchmark_parallel_infer_time.py "${BENCH_ARGS[@]}"
python3 fit_time_model.py
python3 generate_charts.py
python3 summarize_results.py

echo "5.2.15 suite finished."
