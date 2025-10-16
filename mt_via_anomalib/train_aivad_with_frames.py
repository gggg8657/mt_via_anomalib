"""
이미지 프레임을 사용한 AI-VAD 파인튜닝
image_segments.json의 프레임들을 활용합니다.
"""

import os
import json
import torch
from pathlib import Path
from anomalib.data import Folder
from anomalib.models.video import AiVad
from anomalib.engine import Engine
import shutil

def extract_frames_from_json(json_path="image_segments.json", target_dir="frame_dataset"):
    """image_segments.json에서 정상 프레임들을 추출"""
    print(f"📁 이미지 프레임 추출: {json_path}")
    
    # 폴더 생성
    normal_dir = Path(target_dir) / "train" / "good"
    normal_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📂 정상 이미지 폴더: {normal_dir}")
    
    # JSON 파일 로드
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        print(f"  📊 JSON 로드 완료: {len(segments)}개 세그먼트")
    except Exception as e:
        print(f"❌ JSON 로드 실패: {e}")
        return None
    
    # 정상 프레임들 추출
    copied_count = 0
    normal_count = 0
    
    for i, segment in enumerate(segments):
        if segment.get('category') == 'normal' and 'images' in segment:
            normal_count += 1
            images = segment['images']
            
            # 각 세그먼트에서 처음 몇 개 이미지만 사용 (메모리 절약)
            for j, image_path in enumerate(images[:5]):  # 각 세그먼트에서 최대 5개
                if os.path.exists(image_path):
                    # 파일명 생성 (중복 방지)
                    new_name = f"normal_{normal_count:03d}_{j:02d}_{Path(image_path).name}"
                    target_path = normal_dir / new_name
                    
                    try:
                        # 파일 복사
                        shutil.copy2(image_path, target_path)
                        copied_count += 1
                        
                        if copied_count <= 10:  # 처음 10개만 표시
                            print(f"    📸 {new_name}")
                            
                    except Exception as e:
                        print(f"    ⚠️ {image_path} 복사 실패: {e}")
    
    print(f"  ✅ 정상 세그먼트: {normal_count}개")
    print(f"  ✅ 복사된 이미지: {copied_count}개")
    
    if copied_count == 0:
        print("❌ 복사된 이미지가 없습니다!")
        return None
    
    return str(Path(target_dir).absolute())

def main():
    """메인 함수"""
    print("🚀 이미지 프레임을 사용한 AI-VAD 파인튜닝")
    print("=" * 50)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 1. 이미지 프레임 추출
    dataset_root = extract_frames_from_json()
    if dataset_root is None:
        print("❌ 이미지 추출 실패")
        return
    
    # 2. Anomalib Folder 데이터 모듈 생성
    print(f"\n📊 Anomalib Folder 데이터 모듈 생성...")
    try:
        datamodule = Folder(
            name="frame_dataset",
            root=dataset_root,
            normal_dir="train/good",
            train_batch_size=8,  # 이미지라서 배치 크기 증가
            eval_batch_size=8,
            num_workers=0,
            image_size=(256, 256),  # Anomalib 기본 크기
        )
        
        print("✅ Folder 데이터 모듈 생성 완료")
        
    except Exception as e:
        print(f"❌ Folder 데이터 모듈 생성 실패: {e}")
        return
    
    # 3. AI-VAD 모델 생성
    print(f"\n🤖 AI-VAD 모델 생성...")
    try:
        model = AiVad()
        print("✅ AI-VAD 모델 생성 완료")
        
    except Exception as e:
        print(f"❌ AI-VAD 모델 생성 실패: {e}")
        return
    
    # 4. 사전 훈련된 가중치 로드
    checkpoint_path = "aivad_proper_checkpoint.ckpt"
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 사전 훈련된 가중치 로드: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("✅ 사전 훈련된 가중치 로드 완료")
            else:
                print("⚠️ state_dict가 없습니다.")
        except Exception as e:
            print(f"⚠️ 가중치 로드 실패: {e}")
    
    # 5. Anomalib Engine 생성
    print(f"\n🔧 Anomalib Engine 생성...")
    try:
        engine = Engine(
            devices=1 if device == "cuda" else "auto",
            accelerator="gpu" if device == "cuda" else "cpu",
            precision="16-mixed" if device == "cuda" else "32",
            max_epochs=3,  # 빠른 테스트를 위해 적은 에포크
            gradient_clip_val=1.0,
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_model_summary=True,
            # 제한 설정
            limit_train_batches=50,  # 이미지라서 더 많이
            limit_val_batches=20,
        )
        
        print("✅ Engine 생성 완료")
        
    except Exception as e:
        print(f"❌ Engine 생성 실패: {e}")
        return
    
    # 6. 모델 훈련
    print(f"\n🎯 AI-VAD 파인튜닝 시작...")
    try:
        # Anomalib의 공식 훈련 방법
        engine.fit(model=model, datamodule=datamodule)
        
        print("✅ 파인튜닝 완료!")
        
    except Exception as e:
        print(f"❌ 파인튜닝 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 체크포인트 저장
    checkpoint_path = "aivad_frame_finetuned.ckpt"
    try:
        # Anomalib의 공식 저장 방법
        engine.save_checkpoint(checkpoint_path)
        
        print(f"💾 파인튜닝된 모델 저장: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 저장 실패: {e}")
        # 수동 저장
        torch.save({
            'state_dict': model.state_dict(),
            'pytorch-lightning_version': '2.0.0',
            'model_class': 'AiVad',
            'training_type': 'frame_finetuned'
        }, checkpoint_path)
        print(f"💾 수동 저장 완료: {checkpoint_path}")
    
    print("\n🎉 이미지 프레임으로 AI-VAD 파인튜닝 완료!")
    print("💡 다음 단계:")
    print("1. UI에서 'aivad_frame_finetuned.ckpt' 로드")
    print("2. 실제 성능 테스트")
    print("3. 필요시 추가 파인튜닝")

if __name__ == "__main__":
    main()
