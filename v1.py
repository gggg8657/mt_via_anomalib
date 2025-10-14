import argparse, time, math, os
import torch

def human(n):  # pretty print
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

def allocate_reserve(device, reserve_gb, dtype):
    if reserve_gb <= 0:
        return None
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    reserve_bytes = int(reserve_gb * (1024**3))
    numel = reserve_bytes // bytes_per
    print(f"[+] Reserving ~{reserve_gb} GB on {device} as {dtype} "
          f"({human(numel*bytes_per)})")
    # empty_는 실제 값은 안 채우고 메모리만 잡음
    t = torch.empty(numel, dtype=dtype, device=device)
    return t  # 유지하면 해제 안 됨

def do_work(device, matmul_n, dtype, repeats=1):
    # cublas matmul 로 “진짜” 연산 발생시킴 (짧고 굵게)
    a = torch.randn((matmul_n, matmul_n), device=device, dtype=dtype)
    b = torch.randn((matmul_n, matmul_n), device=device, dtype=dtype)
    c = None
    for _ in range(repeats):
        c = a @ b
    # 결과를 사용해 최적화 회피
    s = float(c[0,0].item())  # 작은 동기화 포인트
    return s

def main():
    ap = argparse.ArgumentParser(description="Reserve VRAM + periodic compute")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index")
    ap.add_argument("--reserve-gb", type=float, default=8.0, help="VRAM to reserve")
    ap.add_argument("--matmul-n", type=int, default=2048,
                    help="matmul size (NxN). 2048~4096 추천")
    ap.add_argument("--dtype", choices=["fp16","fp32","bf16"], default="fp16")
    ap.add_argument("--target-util", type=float, default=0.25,
                    help="대략적 목표 연산 점유 비율(0.05~0.9)")
    ap.add_argument("--period-sec", type=float, default=1.0,
                    help="한 주기의 길이(초) — 주기 내에서 work/sleep 분할")
    ap.add_argument("--warmup", type=int, default=3, help="워밍업 반복 수")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    assert torch.cuda.is_available(), "CUDA 사용 불가"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    # 1) 메모리 예약
    reserved = allocate_reserve(device, args.reserve_gb, dtype)

    # 2) 워밍업 (커널 초기화/튜닝)
    print(f"[+] Warmup {args.warmup}x with N={args.matmul_n}, {args.dtype}")
    for _ in range(args.warmup):
        do_work(device, args.matmul_n, dtype, repeats=1)
    torch.cuda.synchronize()

    # 3) 목표 사용률에 맞춰 듀티사이클로 연산
    target_work = max(0.01, min(0.95, args.target_util)) * args.period_sec
    print(f"[+] Start loop: target_util={args.target_util:.2f}, "
          f"work≈{target_work:.3f}s / period={args.period_sec:.3f}s")

    # matmul 1회당 걸리는 시간을 에스티메이트해서 반복 횟수 조절
    # 초기 측정
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    do_work(device, args.matmul_n, dtype, repeats=1)
    end.record(); torch.cuda.synchronize()
    per_call_ms = start.elapsed_time(end) / 1000.0
    if per_call_ms <= 0:
        per_call_ms = 0.01
    print(f"[+] Measured per-call time ≈ {per_call_ms*1000:.1f} ms")

    tick = 0
    while True:
        period_start = time.time()

        # 필요한 반복 횟수 추정
        reps = max(1, int(math.ceil(target_work / per_call_ms)))
        # 과도한 커널폭주 방지: 상한
        reps = min(reps, 1000)

        # 연산
        s = do_work(device, args.matmul_n, dtype, repeats=reps)
        torch.cuda.synchronize()
        work_elapsed = time.time() - period_start

        # sleep으로 목표 사용률 근사
        sleep_time = max(0.0, args.period_sec - work_elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

        tick += 1
        if tick % 10 == 0:
            print(f"[{tick}] worked {work_elapsed:.3f}s, slept {sleep_time:.3f}s, "
                  f"reps={reps}, sample={s:.4f}")

if __name__ == "__main__":
    main()