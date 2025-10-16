"""
Anomalib 공식 방법을 사용한 AI-VAD 파인튜닝
Folder 데이터 모듈과 Engine을 사용합니다.
"""

import os
import torch
from pathlib import Path
from anomalib.data import Folder
from anomalib.models.video import AiVad
from anomalib.engine import Engine
from anomalib.data.utils import TestSplitMode

def prepare_video_folder_structure(video_paths, target_dir="custom_video_dataset"):
    """비디오 파일들을 Anomalib Folder 형식으로 구성"""
    print(f"📁 비디오 폴더 구조 준비: {target_dir}")
    
    # 폴더 생성
    normal_dir = Path(target_dir) / "train" / "good"
    normal_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📂 정상 비디오 폴더: {normal_dir}")
    
    # 비디오 파일들을 정상 폴더로 복사/링크
    copied_count = 0
    for i, video_path in enumerate(video_paths):
        if os.path.exists(video_path):
            # 파일명 변경 (중복 방지)
            new_name = f"normal_{i:03d}_{Path(video_path).name}"
            target_path = normal_dir / new_name
            
            try:
                # 심볼릭 링크 생성 (Windows에서는 복사)
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.copy2(video_path, target_path)
                else:  # Linux/Mac
                    os.symlink(video_path, target_path)
                
                copied_count += 1
                if copied_count <= 5:  # 처음 5개만 표시
                    print(f"    📹 {new_name}")
                    
            except Exception as e:
                print(f"    ⚠️ {video_path} 복사 실패: {e}")
    
    print(f"  ✅ {copied_count}개 비디오 파일 준비 완료")
    return str(Path(target_dir).absolute())

def main():
    """메인 함수"""
    print("🚀 Anomalib 공식 방법으로 AI-VAD 파인튜닝")
    print("=" * 50)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용 디바이스: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # 비디오 파일 경로 설정
    from video_files_list import video_files
    video_paths = video_files
    
    print(f"\n📁 훈련할 비디오 파일: {len(video_paths)}개")
    
    # 1. 비디오 폴더 구조 준비
    dataset_root = prepare_video_folder_structure(video_paths)
    
    # 2. Anomalib Folder 데이터 모듈 생성
    print(f"\n📊 Anomalib Folder 데이터 모듈 생성...")
    try:
        datamodule = Folder(
            name="custom_video_dataset",
            root=dataset_root,
            normal_dir="train/good",
            train_batch_size=2,
            eval_batch_size=2,
            num_workers=0,
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
            limit_train_batches=20,
            limit_val_batches=10,
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
    checkpoint_path = "aivad_official_finetuned.ckpt"
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
            'training_type': 'official_finetuned'
        }, checkpoint_path)
        print(f"💾 수동 저장 완료: {checkpoint_path}")
    
    print("\n🎉 Anomalib 공식 방법으로 파인튜닝 완료!")
    print("💡 다음 단계:")
    print("1. UI에서 'aivad_official_finetuned.ckpt' 로드")
    print("2. 실제 성능 테스트")
    print("3. 필요시 추가 파인튜닝")

if __name__ == "__main__":
    main()
