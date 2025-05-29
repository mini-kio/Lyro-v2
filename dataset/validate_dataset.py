# lyro/dataset/validate_dataset.py
"""
Dataset Validation Script
데이터셋 무결성 검증 및 통계 분석
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import librosa
import torch
from tqdm import tqdm
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import logging

# LYRO 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """데이터셋 검증 클래스"""
    
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.errors = []
        self.warnings = []
        self.statistics = defaultdict(list)
        
    def validate_all(self) -> bool:
        """전체 검증 수행"""
        logger.info("Starting dataset validation...")
        
        # 1. 디렉토리 구조 검증
        self.validate_directory_structure()
        
        # 2. 메타데이터 검증
        self.validate_metadata()
        
        # 3. 오디오 파일 검증
        self.validate_audio_files()
        
        # 4. 가사 파일 검증
        self.validate_lyrics()
        
        # 5. 일관성 검증
        self.validate_consistency()
        
        # 6. 통계 분석
        self.analyze_statistics()
        
        # 결과 출력
        self.print_results()
        
        return len(self.errors) == 0
    
    def validate_directory_structure(self):
        """디렉토리 구조 검증"""
        logger.info("Validating directory structure...")
        
        required_dirs = [
            'metadata',
            'audio/full',
            'lyrics'
        ]
        
        optional_dirs = [
            'audio/instrumental',
            'audio/vocal',
            'processed/latents',
            'processed/phonemes'
        ]
        
        for dir_path in required_dirs:
            full_path = self.dataset_root / dir_path
            if not full_path.exists():
                self.errors.append(f"Required directory missing: {dir_path}")
            elif not full_path.is_dir():
                self.errors.append(f"Path is not a directory: {dir_path}")
                
        for dir_path in optional_dirs:
            full_path = self.dataset_root / dir_path
            if not full_path.exists():
                self.warnings.append(f"Optional directory missing: {dir_path}")
    
    def validate_metadata(self):
        """메타데이터 파일 검증"""
        logger.info("Validating metadata files...")
        
        metadata_files = [
            'metadata/train_metadata.jsonl',
            'metadata/val_metadata.jsonl'
        ]
        
        self.metadata_items = []
        
        for metadata_file in metadata_files:
            file_path = self.dataset_root / metadata_file
            
            if not file_path.exists():
                self.errors.append(f"Metadata file missing: {metadata_file}")
                continue
                
            # 메타데이터 읽기 및 검증
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        self.validate_metadata_item(item, metadata_file, line_num)
                        self.metadata_items.append(item)
                    except json.JSONDecodeError as e:
                        self.errors.append(
                            f"Invalid JSON in {metadata_file} line {line_num}: {e}"
                        )
    
    def validate_metadata_item(self, item: Dict, file_name: str, line_num: int):
        """개별 메타데이터 항목 검증"""
        required_fields = ['id', 'audio_path']
        
        for field in required_fields:
            if field not in item:
                self.errors.append(
                    f"Missing required field '{field}' in {file_name} line {line_num}"
                )
                
        # 파일 경로 존재 확인
        if 'audio_path' in item:
            audio_path = self.dataset_root / item['audio_path']
            if not audio_path.exists():
                self.errors.append(
                    f"Audio file not found: {item['audio_path']} (id: {item.get('id')})"
                )
                
        if 'lyric_path' in item and item['lyric_path']:
            lyric_path = self.dataset_root / item['lyric_path']
            if not lyric_path.exists():
                self.warnings.append(
                    f"Lyric file not found: {item['lyric_path']} (id: {item.get('id')})"
                )
    
    def validate_audio_files(self):
        """오디오 파일 검증"""
        logger.info("Validating audio files...")
        
        audio_dir = self.dataset_root / 'audio' / 'full'
        audio_files = list(audio_dir.glob('*'))
        
        # 샘플링으로 검증 (전체가 너무 많은 경우)
        sample_size = min(len(audio_files), 100)
        sampled_files = np.random.choice(audio_files, sample_size, replace=False)
        
        for audio_file in tqdm(sampled_files, desc="Checking audio files"):
            try:
                # 오디오 로드 시도
                audio, sr = librosa.load(audio_file, sr=None, duration=10)
                
                # 기본 검증
                if sr != 44100:
                    self.warnings.append(
                        f"Non-standard sample rate {sr}Hz: {audio_file.name}"
                    )
                    
                # 통계 수집
                duration = librosa.get_duration(y=audio, sr=sr)
                self.statistics['audio_duration'].append(duration)
                self.statistics['sample_rate'].append(sr)
                
                # 채널 수
                if audio.ndim == 1:
                    channels = 1
                else:
                    channels = audio.shape[0]
                self.statistics['channels'].append(channels)
                
            except Exception as e:
                self.errors.append(f"Failed to load audio {audio_file.name}: {e}")
    
    def validate_lyrics(self):
        """가사 파일 검증"""
        logger.info("Validating lyrics files...")
        
        lyrics_dir = self.dataset_root / 'lyrics'
        if not lyrics_dir.exists():
            return
            
        lyric_files = list(lyrics_dir.glob('*.txt'))
        
        for lyric_file in lyric_files[:100]:  # 샘플링
            try:
                with open(lyric_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 기본 검증
                if not content.strip():
                    self.warnings.append(f"Empty lyric file: {lyric_file.name}")
                    
                # 통계
                self.statistics['lyric_length'].append(len(content))
                self.statistics['lyric_lines'].append(content.count('\n'))
                
            except Exception as e:
                self.errors.append(f"Failed to read lyric file {lyric_file.name}: {e}")
    
    def validate_consistency(self):
        """데이터 일관성 검증"""
        logger.info("Validating data consistency...")
        
        # 메타데이터에 있는 파일들이 실제로 존재하는지
        metadata_audio_paths = set()
        for item in self.metadata_items:
            if 'audio_path' in item:
                metadata_audio_paths.add(item['audio_path'])
                
        # 실제 오디오 파일들
        audio_dir = self.dataset_root / 'audio' / 'full'
        actual_audio_files = set()
        for f in audio_dir.glob('*'):
            if f.is_file():
                rel_path = f.relative_to(self.dataset_root)
                actual_audio_files.add(str(rel_path))
                
        # 차집합 확인
        missing_in_metadata = actual_audio_files - metadata_audio_paths
        missing_in_filesystem = metadata_audio_paths - actual_audio_files
        
        if missing_in_metadata:
            self.warnings.append(
                f"{len(missing_in_metadata)} audio files not in metadata"
            )
            
        if missing_in_filesystem:
            self.errors.append(
                f"{len(missing_in_filesystem)} metadata entries without audio files"
            )
    
    def analyze_statistics(self):
        """통계 분석"""
        logger.info("Analyzing dataset statistics...")
        
        # 장르 분포
        genres = []
        for item in self.metadata_items:
            if 'genre' in item:
                genres.extend(item['genre'])
        self.statistics['genre_distribution'] = Counter(genres)
        
        # 태스크 분포
        task_counts = {'SONG': 0, 'INST': 0, 'COVER': 0}
        for item in self.metadata_items:
            if item.get('lyric_path'):
                task_counts['SONG'] += 1
            else:
                task_counts['INST'] += 1
                
        self.statistics['task_distribution'] = task_counts
    
    def print_results(self):
        """검증 결과 출력"""
        print("\n" + "="*60)
        print("DATASET VALIDATION RESULTS")
        print("="*60)
        
        # 오류
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        else:
            print("\n✅ No errors found!")
            
        # 경고
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings[:10]:
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
                
        # 통계
        print("\n📊 STATISTICS:")
        
        # 오디오 통계
        if self.statistics['audio_duration']:
            durations = self.statistics['audio_duration']
            print(f"\nAudio Duration:")
            print(f"  - Total: {sum(durations)/3600:.2f} hours")
            print(f"  - Average: {np.mean(durations):.2f} seconds")
            print(f"  - Min/Max: {min(durations):.2f} / {max(durations):.2f} seconds")
            
        # 장르 분포
        if self.statistics['genre_distribution']:
            print(f"\nGenre Distribution:")
            for genre, count in self.statistics['genre_distribution'].most_common(10):
                print(f"  - {genre}: {count}")
                
        # 태스크 분포
        if self.statistics['task_distribution']:
            print(f"\nTask Distribution:")
            for task, count in self.statistics['task_distribution'].items():
                print(f"  - {task}: {count}")
                
        print("\n" + "="*60)
    
    def save_report(self, output_path: Path):
        """검증 보고서 저장"""
        report = {
            'errors': self.errors,
            'warnings': self.warnings,
            'statistics': {
                k: v if not isinstance(v, (Counter, defaultdict)) else dict(v)
                for k, v in self.statistics.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Validation report saved to {output_path}")
    
    def plot_statistics(self, output_dir: Path):
        """통계 시각화"""
        output_dir.mkdir(exist_ok=True)
        
        # Duration distribution
        if self.statistics['audio_duration']:
            plt.figure(figsize=(10, 6))
            plt.hist(self.statistics['audio_duration'], bins=50, edgecolor='black')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Count')
            plt.title('Audio Duration Distribution')
            plt.savefig(output_dir / 'duration_distribution.png')
            plt.close()
            
        # Genre distribution
        if self.statistics['genre_distribution']:
            genres = list(self.statistics['genre_distribution'].keys())[:10]
            counts = [self.statistics['genre_distribution'][g] for g in genres]
            
            plt.figure(figsize=(10, 6))
            plt.bar(genres, counts)
            plt.xlabel('Genre')
            plt.ylabel('Count')
            plt.title('Top 10 Genres')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'genre_distribution.png')
            plt.close()


class DatasetFixer:
    """데이터셋 자동 수정 도구"""
    
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.fixes_applied = []
        
    def fix_broken_paths(self, metadata_file: Path):
        """깨진 경로 수정"""
        logger.info(f"Fixing broken paths in {metadata_file}...")
        
        fixed_items = []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                
                # 오디오 경로 수정
                if 'audio_path' in item:
                    audio_path = self.dataset_root / item['audio_path']
                    if not audio_path.exists():
                        # 대체 경로 찾기
                        audio_name = Path(item['audio_path']).name
                        found = list(self.dataset_root.rglob(audio_name))
                        
                        if found:
                            new_path = found[0].relative_to(self.dataset_root)
                            item['audio_path'] = str(new_path)
                            self.fixes_applied.append(
                                f"Fixed audio path: {item['id']}"
                            )
                            
                fixed_items.append(item)
                
        # 백업 생성
        backup_path = metadata_file.with_suffix('.jsonl.backup')
        metadata_file.rename(backup_path)
        
        # 수정된 파일 저장
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for item in fixed_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        logger.info(f"Applied {len(self.fixes_applied)} fixes")


def main():
    parser = argparse.ArgumentParser(description='LYRO Dataset Validation')
    
    parser.add_argument('--dataset_root', type=str, default='dataset/',
                        help='Dataset root directory')
    parser.add_argument('--fix_broken_paths', action='store_true',
                        help='Attempt to fix broken file paths')
    parser.add_argument('--save_report', type=str,
                        help='Save validation report to file')
    parser.add_argument('--plot_stats', action='store_true',
                        help='Generate statistical plots')
    parser.add_argument('--output_dir', type=str, default='validation_results',
                        help='Output directory for reports and plots')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 검증 수행
    validator = DatasetValidator(dataset_root)
    is_valid = validator.validate_all()
    
    # 보고서 저장
    if args.save_report:
        report_path = output_dir / args.save_report
        validator.save_report(report_path)
        
    # 통계 플롯
    if args.plot_stats:
        validator.plot_statistics(output_dir)
        
    # 자동 수정
    if args.fix_broken_paths and not is_valid:
        response = input("\nApply automatic fixes? (y/n): ")
        if response.lower() == 'y':
            fixer = DatasetFixer(dataset_root)
            
            for metadata_file in ['train_metadata.jsonl', 'val_metadata.jsonl']:
                file_path = dataset_root / 'metadata' / metadata_file
                if file_path.exists():
                    fixer.fix_broken_paths(file_path)
                    
    # 종료 코드
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()