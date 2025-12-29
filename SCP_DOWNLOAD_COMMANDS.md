# 📥 SCP 체크포인트 다운로드 명령

## 서버 정보
- **호스트**: `gpu-1` 또는 실제 서버 IP
- **사용자**: `dongjukim`
- **경로**: `/home/dongjukim/workspace/test_via_anomalib`

## 체크포인트 파일
1. **model.ckpt** (761 MB) - Avenue 데이터셋으로 학습된 모델
2. **aivad_extreme_learned.ckpt** (758 MB) - Extreme 학습 모델

## 직접 다운로드 명령 (로컬 PC에서 실행)

### 방법 1: 개별 파일 다운로드

```bash
# Avenue 학습 모델 (권장)
scp dongjukim@gpu-1:/home/dongjukim/workspace/test_via_anomalib/mt_via_anomalib/results/AiVad/Avenue/v1/weights/lightning/model.ckpt ./

# Extreme 학습 모델
scp dongjukim@gpu-1:/home/dongjukim/workspace/test_via_anomalib/aivad_extreme_learned.ckpt ./
```

### 방법 2: 전체 디렉토리 다운로드

```bash
# results 디렉토리 전체 다운로드
scp -r dongjukim@gpu-1:/home/dongjukim/workspace/test_via_anomalib/mt_via_anomalib/results ./

# 또는 체크포인트만 다운로드
mkdir -p ./checkpoints
scp dongjukim@gpu-1:/home/dongjukim/workspace/test_via_anomalib/mt_via_anomalib/results/AiVad/Avenue/v1/weights/lightning/model.ckpt ./checkpoints/
scp dongjukim@gpu-1:/home/dongjukim/workspace/test_via_anomalib/aivad_extreme_learned.ckpt ./checkpoints/
```

### 방법 3: 압축 후 다운로드 (더 빠름)

```bash
# 서버에서 실행 (압축)
ssh dongjukim@gpu-1 "cd /home/dongjukim/workspace/test_via_anomalib && tar czf checkpoints.tar.gz mt_via_anomalib/results/AiVad/Avenue/v1/weights/lightning/model.ckpt aivad_extreme_learned.ckpt"

# 로컬에서 다운로드 (압축 파일)
scp dongjukim@gpu-1:/home/dongjukim/workspace/test_via_anomalib/checkpoints.tar.gz ./

# 로컬에서 압축 해제
tar xzf checkpoints.tar.gz
```

## 스크립트 사용 (자동화)

```bash
# 서버에서 스크립트 다운로드
scp dongjukim@gpu-1:/home/dongjukim/workspace/test_via_anomalib/download_checkpoints.sh ./

# 로컬에서 실행 (서버 정보 수정 필요)
bash download_checkpoints.sh
```

## 빠른 다운로드 팁

1. **병렬 다운로드**: 여러 터미널에서 동시에 다운로드
2. **압축 사용**: tar.gz로 압축 후 전송 (더 빠름)
3. **rsync 사용**: 중단 시 재개 가능
   ```bash
   rsync -avzP dongjukim@gpu-1:/home/dongjukim/workspace/test_via_anomalib/mt_via_anomalib/results/AiVad/Avenue/v1/weights/lightning/model.ckpt ./
   ```

## 파일 정보

- **model.ckpt**: 760.7 MB - Avenue 데이터셋으로 학습 완료 ✅
- **aivad_extreme_learned.ckpt**: 757.9 MB - Extreme 학습 완료

