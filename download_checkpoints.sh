#!/bin/bash
# 체크포인트 파일 다운로드 스크립트
# 사용법: 로컬 PC에서 실행
# bash download_checkpoints.sh

echo "=========================================="
echo "📥 AiVAD 체크포인트 다운로드"
echo "=========================================="
echo ""

# 서버 정보 (수정 필요)
SERVER_USER="dongjukim"
SERVER_HOST="$(hostname)"  # 실제 서버 호스트명으로 변경하세요
SERVER_DIR="/home/dongjukim/workspace/test_via_anomalib"

# 로컬 저장 경로 (수정 필요)
LOCAL_DIR="./checkpoints"

# 체크포인트 파일들
CHECKPOINTS=(
    "mt_via_anomalib/results/AiVad/Avenue/v1/weights/lightning/model.ckpt"
    "aivad_extreme_learned.ckpt"
)

# 로컬 디렉토리 생성
mkdir -p "$LOCAL_DIR"

echo "📁 로컬 저장 경로: $LOCAL_DIR"
echo "🖥️  서버: ${SERVER_USER}@${SERVER_HOST}"
echo "📂 서버 경로: $SERVER_DIR"
echo ""

# 각 체크포인트 다운로드
for ckpt in "${CHECKPOINTS[@]}"; do
    echo "⬇️  다운로드 중: $ckpt"
    scp "${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}/${ckpt}" "${LOCAL_DIR}/"
    
    if [ $? -eq 0 ]; then
        echo "✅ 다운로드 완료: $ckpt"
        ls -lh "${LOCAL_DIR}/$(basename $ckpt)"
    else
        echo "❌ 다운로드 실패: $ckpt"
    fi
    echo ""
done

echo "=========================================="
echo "🎉 다운로드 완료!"
echo "=========================================="
echo ""
echo "📊 다운로드된 파일:"
ls -lh "$LOCAL_DIR"/*.ckpt 2>/dev/null || echo "파일 없음"

