"""
UI 호환 체크포인트 생성 스크립트
realtime_ui_advanced_windows.py에서 사용할 수 있는 형식으로 체크포인트 생성
"""

import os
import torch
import pytorch_lightning as pl
from anomalib.models.video import AiVad

# GPU 설정
print("🔧 GPU 설정 중...")
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.empty_cache()
    print(f"✅ GPU 사용: {torch.cuda.get_device_name()}")
else:
    device = "cpu"
    print("⚠️  CPU 사용")

def create_ui_compatible_checkpoint():
    """UI 호환 체크포인트 생성"""
    print("🤖 AI-VAD 모델 초기화...")
    
    try:
        # 모델 초기화
        model = AiVad()
        
        # GPU로 이동
        if device == "cuda":
            model = model.cuda()
            print("✅ 모델을 GPU로 이동 완료")
        
        # Lightning 체크포인트 형식으로 저장
        checkpoint_path = "aivad_ui_compatible_checkpoint.ckpt"
        
        # Lightning 모델을 체크포인트로 저장
        pl.save_checkpoint(model, checkpoint_path)
        
        print(f"💾 UI 호환 체크포인트 저장: {checkpoint_path}")
        
        # 파일 크기 확인
        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
            print(f"📊 체크포인트 크기: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 체크포인트 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_checkpoint_loading():
    """체크포인트 로딩 테스트"""
    print("🧪 체크포인트 로딩 테스트...")
    
    try:
        checkpoint_path = "aivad_ui_compatible_checkpoint.ckpt"
        
        if not os.path.exists(checkpoint_path):
            print("❌ 체크포인트 파일이 존재하지 않습니다.")
            return False
        
        # 체크포인트 로드 테스트
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("✅ 체크포인트 로드 성공")
        
        # 필요한 키 확인
        required_keys = ['state_dict', 'pytorch-lightning_version']
        for key in required_keys:
            if key in checkpoint:
                print(f"✅ {key} 키 존재")
            else:
                print(f"❌ {key} 키 없음")
        
        return True
        
    except Exception as e:
        print(f"❌ 체크포인트 로딩 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🏆 UI 호환 체크포인트 생성")
    print("=" * 60)
    
    # 체크포인트 생성
    success = create_ui_compatible_checkpoint()
    
    if success:
        # 로딩 테스트
        test_success = test_checkpoint_loading()
        
        if test_success:
            print("\n🎉 UI 호환 체크포인트 생성 완료!")
            print("\n📁 생성된 파일:")
            print("- aivad_ui_compatible_checkpoint.ckpt")
            print("\n🚀 사용 방법:")
            print("1. python realtime_ui_advanced_windows.py 실행")
            print("2. '체크포인트 로드' 버튼 클릭")
            print("3. aivad_ui_compatible_checkpoint.ckpt 파일 선택")
            print("4. 실시간 이상 탐지 테스트")
        else:
            print("\n⚠️  체크포인트는 생성되었지만 로딩 테스트 실패")
    else:
        print("\n💥 체크포인트 생성 실패")
        exit(1)
