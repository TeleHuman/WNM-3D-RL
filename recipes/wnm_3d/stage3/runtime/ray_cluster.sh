#!/usr/bin/env bash
# Stage-3 official arbitrary-node Ray cluster bootstrap.
set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export OPENCV_FFMPEG_THREADS=1
export WAM_ACTION_BACKBONE_GRAD_GAIN=${WAM_ACTION_BACKBONE_GRAD_GAIN:-1.0}

# Ray/Gloo bootstrap over the routable management bond; NCCL payloads use
# all eight 400-Gbit InfiniBand HCAs explicitly.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
# The leading '=' requests exact NCCL matching.  Without it, mlx5_1 also
# matches the mlx5_14/15 RoCE management devices and can exhaust CQ resources.
export NCCL_IB_HCA=${NCCL_IB_HCA:-'=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7'}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export CPUS_PER_NODE=${CPUS_PER_NODE:-32}
export NNODES=${NNODES:-1}
export RAY_PORT=${RAY_PORT:-6379}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
export RAY_OBJECT_STORE_BYTES=${RAY_OBJECT_STORE_BYTES:-42949672960}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:?Set MASTER_ADDR to the Ray head address}
export RAY_NODE_IFNAME=${RAY_NODE_IFNAME:-bond0}
RAY_BIN=${RAY_BIN:-ray}
PYTHON_BIN=${PYTHON_BIN:-python}

# Prefer an explicit address. Otherwise use iproute2 when available, with a
# Python/Linux ioctl fallback for minimal training images that do not ship the
# `ip` executable. Both automatic paths resolve the requested interface rather
# than guessing from the host's first address.
resolve_node_ipv4() {
    if command -v ip >/dev/null 2>&1; then
        ip -4 -o addr show dev "$RAY_NODE_IFNAME" \
            | awk '{split($4, a, "/"); print a[1]; exit}'
        return
    fi
    "$PYTHON_BIN" - "$RAY_NODE_IFNAME" <<'PY'
import fcntl
import socket
import struct
import sys

interface = sys.argv[1].encode("utf-8")
if not interface or len(interface) > 15:
    raise SystemExit(0)

try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        request = struct.pack("256s", interface)
        response = fcntl.ioctl(sock.fileno(), 0x8915, request)  # SIOCGIFADDR
    print(socket.inet_ntoa(response[20:24]))
except OSError:
    pass
PY
}

export NODE_IP=${NODE_IP:-}
if [[ -z "$NODE_IP" ]]; then
    NODE_IP=$(resolve_node_ipv4)
    export NODE_IP
fi
if [[ -z "$NODE_IP" ]]; then
    echo "No IPv4 address found on RAY_NODE_IFNAME=$RAY_NODE_IFNAME; set NODE_IP explicitly or fix the interface name" >&2
    exit 2
fi

RUNTIME_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${WAM_REPO_ROOT:-$(cd "$RUNTIME_DIR/../../../.." && pwd)}
WNM3D_SOURCE_DIR=${WNM3D_SOURCE_ROOT:?Set WNM3D_SOURCE_ROOT in paths.env}
export WAM_REPO_ROOT=$REPO_ROOT
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$RUNTIME_DIR/train_entry.sh}
# Ray workers are spawned by the pre-existing raylets, before the head-side
# training wrapper runs.  Put the model package on both raylets' import path
# here instead of relying on the later driver-only export.
export PYTHONPATH=$WNM3D_SOURCE_DIR:$REPO_ROOT:${PYTHONPATH:-}

# GN0 deployment sampler semantics. These must be present before each raylet
# starts so workers on both nodes override the checkpoint's training-only
# num_inference_timesteps=4 with the deployed 8/16 schedule.
export NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-16}
export ENABLE_DIT_CACHE=${ENABLE_DIT_CACHE:-true}
export ENABLE_CFG=${ENABLE_CFG:-true}
export CFG_SCALE=${CFG_SCALE:-5.0}
export DYNAMIC_CACHE_SCHEDULE=${DYNAMIC_CACHE_SCHEDULE:-false}

ulimit -u 65535 || true
ulimit -n 65535 || true

if [[ "${WAM_VERIFY_EIGHT_HCA_TOPOLOGY:-true}" == "true" ]]; then
    for index in 0 1 2 3 4 5 6 7; do
        state_file=/sys/class/infiniband/mlx5_${index}/ports/1/state
        if [[ ! -r "$state_file" ]]; then
            echo "Missing expected InfiniBand state file: $state_file" >&2
            exit 2
        fi
        state=$(<"$state_file")
        if [[ "$state" != *ACTIVE* ]]; then
            echo "mlx5_${index} is not ACTIVE: $state" >&2
            exit 2
        fi
    done
fi

echo "node_rank=$NODE_RANK node_ip=$NODE_IP interface=$RAY_NODE_IFNAME master=$MASTER_ADDR nodes=$NNODES GPUs/node=$GPUS_PER_NODE"
echo "NCCL socket=$NCCL_SOCKET_IFNAME IB_HCA=$NCCL_IB_HCA IB_DISABLE=$NCCL_IB_DISABLE"
echo "WNM sampler: steps=$NUM_INFERENCE_STEPS dit_cache=$ENABLE_DIT_CACHE cfg=$ENABLE_CFG cfg_scale=$CFG_SCALE dynamic_cache=$DYNAMIC_CACHE_SCHEDULE"

case "${WAM_REPLACE_EXISTING_RAY:-false}" in
    true)
        echo "WAM_REPLACE_EXISTING_RAY=true: stopping the node-local Ray runtime"
        "$RAY_BIN" stop --force
        ;;
    false)
        # ``ray status --address=auto`` can wait indefinitely when no local
        # GCS exists. A live node-local runtime always owns a raylet process,
        # so inspect that process directly and fail closed without contacting
        # a possibly absent cluster.
        if pgrep -x raylet >/dev/null 2>&1; then
            echo "A Ray runtime is already active on this node; refusing to stop it implicitly." >&2
            exit 2
        fi
        ;;
    *)
        echo "WAM_REPLACE_EXISTING_RAY must be true or false" >&2
        exit 2
        ;;
esac

if [[ "$NODE_RANK" == "0" ]]; then
    export RAY_ADDRESS=auto
    "$RAY_BIN" start --head \
        --node-ip-address="$NODE_IP" \
        --port="$RAY_PORT" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="$RAY_DASHBOARD_PORT" \
        --num-gpus="$GPUS_PER_NODE" \
        --num-cpus="$CPUS_PER_NODE" \
        --object-store-memory="$RAY_OBJECT_STORE_BYTES" \
        --disable-usage-stats

    ready_nodes=0
    for _ in $(seq 1 180); do
        ready_nodes=$("$PYTHON_BIN" - <<'PY'
import ray
ray.init(address="auto", logging_level="ERROR")
print(sum(bool(node.get("Alive")) for node in ray.nodes()))
ray.shutdown()
PY
        )
        echo "Ray nodes ready: $ready_nodes/$NNODES"
        (( ready_nodes >= NNODES )) && break
        sleep 5
    done
    if (( ready_nodes < NNODES )); then
        echo "Ray worker did not join in time" >&2
        exit 3
    fi

    "$PYTHON_BIN" - <<'PY'
import ray
ray.init(address="auto", logging_level="ERROR")
print("cluster_resources", ray.cluster_resources())
for node in ray.nodes():
    if node.get("Alive"):
        print("alive_node", node["NodeManagerAddress"], node["Resources"])
ray.shutdown()
PY
    cd "$REPO_ROOT"
    exec bash "$TRAIN_SCRIPT" "$@"
fi

for _ in $(seq 1 180); do
    if timeout 3 bash -c "</dev/tcp/$MASTER_ADDR/$RAY_PORT" 2>/dev/null; then
        break
    fi
    sleep 5
done

"$RAY_BIN" start \
    --address="$MASTER_ADDR:$RAY_PORT" \
    --node-ip-address="$NODE_IP" \
    --num-gpus="$GPUS_PER_NODE" \
    --num-cpus="$CPUS_PER_NODE" \
    --object-store-memory="$RAY_OBJECT_STORE_BYTES" \
    --disable-usage-stats

while "$RAY_BIN" status --address="$MASTER_ADDR:$RAY_PORT" >/dev/null 2>&1; do
    sleep 60
done
