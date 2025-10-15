"""
AI-VAD 모델 구조 확인 스크립트
현재 로드된 모델이 실제 AI-VAD 구조인지 확인
"""

import torch
import os
from anomalib.models.video import AiVad

def check_model_structure():
    """모델 구조 확인"""
    print("🔍 AI-VAD 모델 구조 확인 중...")
    
    try:
        # 새로운 AI-VAD 모델 생성
        model = AiVad()
        print("✅ AI-VAD 모델 생성 성공")
        
        # 모델 구조 출력
        print("\n📊 모델 구조:")
        print(model)
        
        # 모델 컴포넌트 확인
        print("\n🔍 모델 컴포넌트 확인:")
        if hasattr(model, 'model'):
            core_model = model.model
            print(f"✅ Core 모델 타입: {type(core_model)}")
            
            # AI-VAD 특정 컴포넌트 확인
            components = ['flow_extractor', 'region_extractor', 'clip_extractor', 'feature_extractor']
            for comp in components:
                if hasattr(core_model, comp):
                    print(f"✅ {comp}: {type(getattr(core_model, comp))}")
                else:
                    print(f"❌ {comp}: 없음")
        
        # 체크포인트 파일 확인
        checkpoint_path = "aivad_ui_ready_checkpoint.ckpt"
        if os.path.exists(checkpoint_path):
            print(f"\n📁 체크포인트 파일 확인: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            print(f"📊 체크포인트 키들: {list(checkpoint.keys())}")
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"📊 State dict 키 개수: {len(state_dict)}")
                
                # 중요한 키들 확인
                important_keys = ['flow', 'clip', 'region', 'feature']
                found_keys = []
                for key in state_dict.keys():
                    for imp_key in important_keys:
                        if imp_key in key.lower():
                            found_keys.append(key)
                
                print(f"🔍 AI-VAD 관련 키들:")
                for key in found_keys[:10]:  # 처음 10개만 표시
                    print(f"   - {key}: {state_dict[key].shape}")
                
                if len(found_keys) == 0:
                    print("⚠️  AI-VAD 관련 키를 찾을 수 없습니다!")
                    print("   이는 단순한 더미 모델일 가능성이 높습니다.")
                
        else:
            print(f"❌ 체크포인트 파일이 존재하지 않습니다: {checkpoint_path}")
            
    except Exception as e:
        print(f"❌ 모델 구조 확인 실패: {e}")
        import traceback
        traceback.print_exc()

def check_anomalib_version():
    """Anomalib 버전 확인"""
    try:
        import anomalib
        print(f"📊 Anomalib 버전: {anomalib.__version__}")
        
        # AI-VAD 관련 클래스 확인
        from anomalib.models.video.ai_vad import AiVad
        print(f"✅ AI-VAD 클래스: {AiVad}")
        
    except Exception as e:
        print(f"❌ Anomalib 버전 확인 실패: {e}")

def create_proper_aivad_checkpoint():
    """올바른 AI-VAD 체크포인트 생성"""
    print("\n🏗️  올바른 AI-VAD 체크포인트 생성 중...")
    
    try:
        # AI-VAD 모델 생성
        model = AiVad()
        print("✅ AI-VAD 모델 생성 완료")
        
        # 모델 컴포넌트 확인
        if hasattr(model, 'model'):
            core = model.model
            print(f"✅ Core 모델: {type(core)}")
            
            # AI-VAD 컴포넌트들 확인
            if hasattr(core, 'flow_extractor'):
                print("✅ Flow extractor 존재")
            if hasattr(core, 'region_extractor'):
                print("✅ Region extractor 존재")
            if hasattr(core, 'clip_extractor'):
                print("✅ CLIP extractor 존재")
            else:
                print("⚠️  CLIP extractor 없음 - 이는 문제가 될 수 있습니다!")
        
        # 체크포인트 저장
        checkpoint_path = "aivad_proper_checkpoint.ckpt"
        checkpoint = {
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.5.5',
            'model_class': 'AiVad',
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 올바른 체크포인트 저장: {checkpoint_path}")
        
        # 파일 크기 확인
        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
            print(f"📊 체크포인트 크기: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 올바른 체크포인트 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 AI-VAD 모델 구조 확인")
    print("=" * 60)
    
    # Anomalib 버전 확인
    check_anomalib_version()
    
    # 모델 구조 확인
    check_model_structure()
    
    # 올바른 체크포인트 생성
    success = create_proper_aivad_checkpoint()
    
    if success:
        print("\n🎉 올바른 AI-VAD 체크포인트 생성 완료!")
        print("\n💡 다음 단계:")
        print("1. UI에서 'aivad_proper_checkpoint.ckpt' 로드")
        print("2. 실제 AI-VAD 구조로 추론 테스트")
        print("3. CLIP 등의 컴포넌트 포함 여부 확인")
    else:
        print("\n💥 체크포인트 생성 실패")
        print("\n💡 해결 방법:")
        print("1. Anomalib 버전 확인")
        print("2. AI-VAD 설치 상태 확인")
        print("3. 의존성 패키지 설치 확인")
