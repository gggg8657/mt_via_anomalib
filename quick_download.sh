#!/bin/bash
# 빠른 다운로드를 위한 압축 스크립트

echo "📦 체크포인트 파일 압축 중..."

cd /home/dongjukim/workspace/test_via_anomalib

# 체크포인트 파일 압축
tar czf checkpoints.tar.gz \
    mt_via_anomalib/results/AiVad/Avenue/v1/weights/lightning/model.ckpt \
    aivad_extreme_learned.ckpt

if [ $? -eq 0 ]; then
    echo "✅ 압축 완료: checkpoints.tar.gz"
    ls -lh checkpoints.tar.gz
    echo ""
    echo "📥 로컬 PC에서 다운로드 명령:"
    echo "scp dongjukim@gpu-1:$(pwd)/checkpoints.tar.gz ./"
else
    echo "❌ 압축 실패"
fi
