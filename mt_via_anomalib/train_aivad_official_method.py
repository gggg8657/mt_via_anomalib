"""
Accurate Interpretable VAD (AiVAD) 공식 학습 방법
test.ipynb에서 성공한 방법을 따라 구현
"""

import os
import torch
from pathlib import Path
from anomalib.models.video import AiVad
from anomalib.data import Avenue
from anomalib.engine import Engine
from anomalib.data.datasets.base.video import VideoTargetFrame

# cuDNN 설정 (test.ipynb에서 사용한 설정)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('medium')

# pandas 버전 호환성 패치
def patch_avenue_dataset():
    """Avenue 데이터셋의 pandas 버전 호환성 문제 패치"""
    try:
        from anomalib.data.datasets.video import avenue
        import pandas as pd
        from pathlib import Path
        
        original_func = avenue.make_avenue_dataset
        
        # 직접 구현 방식으로 Avenue 데이터셋 로딩 (pandas 문제 완전 우회)
        def patched_make_avenue_dataset(root, gt_dir, split):
            # anomalib의 원본 로직을 수동으로 구현 (pandas 문제 수정)
            root = Path(root)
            gt_dir = Path(gt_dir) if gt_dir else None
            
            # Avenue 데이터셋 파일 찾기
            samples_list = []
            training_dir = root / "training_videos"
            testing_dir = root / "testing_videos"
            
            if training_dir.exists():
                for video_file in sorted(training_dir.glob("*.avi")):
                    samples_list.append({
                        'image_path': str(video_file),
                        'video_path': str(video_file),
                        'folder': 'training_videos',
                        'split': 'train',
                        'mask_path': '',
                        'root': str(root),
                    })
            
            if testing_dir.exists():
                for video_file in sorted(testing_dir.glob("*.avi")):
                    mask_path = ''
                    if gt_dir and (gt_dir / "testing_label_mask" / video_file.name).exists():
                        mask_path = str(gt_dir / "testing_label_mask" / video_file.name)
                    
                    samples_list.append({
                        'image_path': str(video_file),
                        'video_path': str(video_file),
                        'folder': 'testing_videos',
                        'split': 'test',
                        'mask_path': mask_path,
                        'root': str(root),
                    })
            
            # DataFrame 생성 (명시적 인덱스 설정)
            if samples_list:
                samples = pd.DataFrame(samples_list)
                samples = samples.reset_index(drop=True)
            else:
                samples = pd.DataFrame(columns=['image_path', 'video_path', 'folder', 'split', 'mask_path', 'root'])
            
            # split 필터링
            if split:
                samples = samples[samples.split == split].reset_index(drop=True)
            
            return samples
        
        avenue.make_avenue_dataset = patched_make_avenue_dataset
        print("✅ Avenue 데이터셋 pandas 패치 적용 완료")
        return True
    except Exception as e:
        print(f"⚠️ pandas 패치 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """AiVAD 공식 학습 방법 (test.ipynb 기반)"""
    print("=" * 60)
    print("🚀 Accurate Interpretable VAD (AiVAD) 공식 학습 시작")
    print("💡 test.ipynb에서 성공한 방법 사용")
    print("=" * 60)
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ 사용 디바이스: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # pandas 버전 호환성 패치 적용
    patch_avenue_dataset()
    
    # 1. Avenue 데이터셋 준비 (test.ipynb와 동일한 방법)
    print("\n📁 Avenue 데이터셋 준비 중...")
    print("💡 Avenue 데이터셋은 자동으로 다운로드됩니다.")
    
    try:
        datamodule = Avenue(
            root="./data/anomalib",  # 데이터 저장 경로
            clip_length_in_frames=2,  # AiVAD는 2프레임 클립 사용
            frames_between_clips=1,   # 클립 간 1프레임 간격
            target_frame=VideoTargetFrame.LAST,  # 마지막 프레임 타겟
            num_workers=2,            # test.ipynb와 동일
        )
        print("✅ Avenue 데이터 모듈 생성 완료")
        
        # 데이터 다운로드 (필요시)
        print("📥 데이터셋 다운로드 확인 중...")
        datamodule.prepare_data()
        print("✅ 데이터셋 준비 완료")
        
    except Exception as e:
        print(f"❌ 데이터셋 준비 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. AiVAD 모델 초기화 (test.ipynb와 동일)
    print("\n🤖 AiVAD 모델 초기화...")
    try:
        model = AiVad()  # 기본 설정 사용 (test.ipynb와 동일)
        print("✅ AiVAD 모델 생성 완료")
        
    except Exception as e:
        print(f"❌ 모델 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 학습 엔진 설정 (test.ipynb와 동일한 설정)
    print("\n🔧 PyTorch Lightning Engine 설정 (test.ipynb 방식)...")
    try:
        engine = Engine(
            devices=1,  # 단일 GPU 사용 (test.ipynb와 동일)
            accelerator='gpu' if device == "cuda" else "cpu",
            precision='32',  # 32-bit precision (test.ipynb와 동일, cuDNN 호환성)
            max_epochs=10,  # test.ipynb와 동일
            limit_train_batches=5,  # test.ipynb와 동일 (메모리 사용량 감소)
            limit_val_batches=2,  # test.ipynb와 동일
            accumulate_grad_batches=1,  # test.ipynb와 동일
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        print("✅ Engine 설정 완료")
        
    except Exception as e:
        print(f"❌ Engine 설정 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 학습 시작 (test.ipynb와 동일한 방법)
    print("\n🎯 AiVAD 학습 시작!")
    print("💡 학습 과정:")
    print("   1. 정상 비디오 클립으로 Feature Extraction (Flow, Region, Pose, Deep)")
    print("   2. Density Estimator로 정상 데이터의 분포 학습")
    print("   3. One-Class Learning (정상 데이터만 사용)")
    print("   4. 이상 탐지를 위한 분포 모델 생성")
    print()
    
    try:
        # test.ipynb와 동일한 방법: engine.fit() 사용
        engine.fit(model=model, datamodule=datamodule)
        
        print("\n✅ 학습 완료!")
        
    except Exception as e:
        print(f"\n❌ 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 체크포인트 저장
    checkpoint_path = "aivad_official_trained.ckpt"
    print(f"\n💾 체크포인트 저장 중: {checkpoint_path}")
    
    try:
        # PyTorch Lightning의 공식 저장 방법
        if hasattr(engine.trainer, 'save_checkpoint'):
            engine.trainer.save_checkpoint(checkpoint_path)
        else:
            # 수동 저장 (백업)
            torch.save({
                'state_dict': model.state_dict(),
                'hyper_parameters': model.hparams if hasattr(model, 'hparams') else {},
                'pytorch-lightning_version': '2.0.0',
                'model_class': 'AiVad',
                'training_type': 'official_anomalib_method',
            }, checkpoint_path)
        
        print(f"✅ 체크포인트 저장 완료: {checkpoint_path}")
        print(f"📊 파일 크기: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"⚠️ 체크포인트 저장 실패: {e}")
        # 최소한의 저장 시도
        try:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ 최소 저장 완료 (state_dict만): {checkpoint_path}")
        except Exception as e2:
            print(f"❌ 최소 저장도 실패: {e2}")
    
    print("\n" + "=" * 60)
    print("🎉 AiVAD 공식 학습 완료!")
    print("=" * 60)
    print(f"\n💡 다음 단계:")
    print(f"1. UI에서 '{checkpoint_path}' 파일을 로드하여 사용")
    print(f"2. 이상 탐지 테스트 수행")
    print(f"3. 필요시 추가 파인튜닝")
    print()

if __name__ == "__main__":
    main()
