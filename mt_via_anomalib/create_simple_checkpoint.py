"""
간단한 UI 호환 체크포인트 생성 (PyTorch 2.6 호환)
복잡한 Lightning 구조 없이 간단하게 생성
"""

import os
import torch
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

def create_simple_checkpoint():
    """간단한 UI 호환 체크포인트 생성"""
    print("🤖 AI-VAD 모델 초기화...")
    
    try:
        # 모델 초기화
        model = AiVad()
        
        # GPU로 이동
        if device == "cuda":
            model = model.cuda()
            print("✅ 모델을 GPU로 이동 완료")
        
        # 간단한 체크포인트 형식으로 저장
        checkpoint_path = "aivad_simple_checkpoint.ckpt"
        
        # 기본적인 체크포인트 구조만 사용
        checkpoint = {
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.5.5',  # 고정 버전
            'model_class': 'AiVad',
        }
        
        # 간단한 저장 (복잡한 구조 없이)
        torch.save(checkpoint, checkpoint_path)
        
        print(f"💾 간단한 UI 호환 체크포인트 저장: {checkpoint_path}")
        
        # 파일 크기 확인
        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
            print(f"📊 체크포인트 크기: {size_mb:.1f} MB")
        
        # 간단한 로드 테스트
        print("🧪 체크포인트 로드 테스트...")
        try:
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            print("✅ 체크포인트 로드 성공")
            
            # 키 확인
            for key in loaded_checkpoint.keys():
                print(f"   - {key}: {type(loaded_checkpoint[key])}")
                
        except Exception as load_error:
            print(f"❌ 체크포인트 로드 테스트 실패: {load_error}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 체크포인트 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🏆 간단한 UI 호환 체크포인트 생성 (PyTorch 2.6 호환)")
    print("=" * 60)
    
    success = create_simple_checkpoint()
    
    if success:
        print("\n🎉 간단한 UI 호환 체크포인트 생성 완료!")
        print("\n📁 생성된 파일:")
        print("- aivad_simple_checkpoint.ckpt")
        print("\n🚀 사용 방법:")
        print("1. python realtime_ui_advanced_windows.py 실행")
        print("2. '체크포인트 로드' 버튼 클릭")
        print("3. aivad_simple_checkpoint.ckpt 파일 선택")
        print("4. 실시간 이상 탐지 테스트")
        print("\n💡 이 버전의 특징:")
        print("- PyTorch 2.6 완전 호환")
        print("- 간단한 체크포인트 구조")
        print("- 복잡한 Lightning 구조 제거")
        print("- 안정적인 로드/저장")
    else:
        print("\n💥 체크포인트 생성 실패")
        exit(1)
