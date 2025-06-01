# lyro/dataset/validate_dataset.py
"""
Dataset Validation Script with TTS Support
데이터셋 무결성 검증 및 통계 분석 (TTS 포함)
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
    """데이터셋 검증 클래스 with TTS support"""
    
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.errors = []
        self.warnings = []
        self.statistics = defaultdict(list)
        
    def validate_all(self) -> bool:
        """전체 검증 수행"""
        logger.info("Starting dataset validation with TTS support...")
        
        # 1. 디렉토리 구조 검증
        self.validate_directory_structure()
        
        # 2. 메타데이터 검증
        self.validate_metadata()
        
        # 3. 오디오 파일 검증
        self.validate_audio_files()
        
        # 4. 가사 파일 검증
        self.validate_lyrics()
        
        # 5. TTS 파일 검증
        self.validate_tts_files()
        
        # 6. 전사본 파일 검증
        self.validate_transcripts()
        
        # 7. 일관성 검증
        self.validate_consistency()
        
        # 8. 통계 분석
        self.analyze_statistics()
        
        # 결과 출력
        self.print_results()
        
        return len(self.errors) == 0
    
    def validate_directory_structure(self):
        """디렉토리 구조 검증 with TTS"""
        logger.info("Validating directory structure...")
        
        required_dirs = [
            'metadata',
            'audio/full',
            'lyrics'
        ]
        
        optional_dirs = [
            'audio/tts',           # TTS 오디오
            'transcript',          # TTS 전사본
            'audio/instrumental',
            'audio/vocal',
            'processed/latents',
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
        """메타데이터 파일 검증 with TTS"""
        logger.info("Validating metadata files...")
        
        metadata_files = [
            'metadata/train_metadata.jsonl',
            'metadata/val_metadata.jsonl',
            'metadata/tts_metadata.jsonl'  # TTS 메타데이터 추가
        ]
        
        self.metadata_items = []
        
        for metadata_file in metadata_files:
            file_path = self.dataset_root / metadata_file
            
            if not file_path.exists():
                if 'tts' in metadata_file:
                    self.warnings.append(f"TTS metadata file missing: {metadata_file}")
                else:
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
        """개별 메타데이터 항목 검증 with TTS"""
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
                
        # 가사 파일 확인
        if 'lyric_path' in item and item['lyric_path']:
            lyric_path = self.dataset_root / item['lyric_path']
            if not lyric_path.exists():
                self.warnings.append(
                    f"Lyric file not found: {item['lyric_path']} (id: {item.get('id')})"
                )
        
        # TTS 관련 파일 확인
        if item.get('has_tts', False):
            if 'tts_audio_path' in item and item['tts_audio_path']:
                tts_audio_path = self.dataset_root / item['tts_audio_path']
                if not tts_audio_path.exists():
                    self.warnings.append(
                        f"TTS audio file not found: {item['tts_audio_path']} (id: {item.get('id')})"
                    )
            
            if 'transcript_path' in item and item['transcript_path']:
                transcript_path = self.dataset_root / item['transcript_path']
                if not transcript_path.exists():
                    self.warnings.append(
                        f"Transcript file not found: {item['transcript_path']} (id: {item.get('id')})"
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
    
    def validate_tts_files(self):
        """TTS 오디오 파일 검증"""
        logger.info("Validating TTS audio files...")
        
        tts_dir = self.dataset_root / 'audio' / 'tts'
        if not tts_dir.exists():
            self.warnings.append("TTS audio directory not found")
            return
            
        tts_files = list(tts_dir.glob('*'))
        
        # 샘플링으로 검증
        sample_size = min(len(tts_files), 100)
        if len(tts_files) > 0:
            sampled_files = np.random.choice(tts_files, sample_size, replace=False)
            
            for tts_file in tqdm(sampled_files, desc="Checking TTS files"):
                try:
                    # TTS 오디오 로드 시도
                    audio, sr = librosa.load(tts_file, sr=None, duration=10)
                    
                    # TTS 특화 검증
                    if sr != 44100:
                        self.warnings.append(
                            f"TTS non-standard sample rate {sr}Hz: {tts_file.name}"
                        )
                        
                    # 통계 수집 (TTS 별도)
                    duration = librosa.get_duration(y=audio, sr=sr)
                    self.statistics['tts_duration'].append(duration)
                    
                    # TTS는 일반적으로 모노
                    if audio.ndim == 1:
                        channels = 1
                    else:
                        channels = audio.shape[0]
                    self.statistics['tts_channels'].append(channels)
                    
                    # TTS 음성 특성 검증
                    # 너무 조용하거나 시끄러운지 확인
                    rms = np.sqrt(np.mean(audio ** 2))
                    if rms < 0.001:
                        self.warnings.append(f"TTS file too quiet: {tts_file.name}")
                    elif rms > 0.5:
                        self.warnings.append(f"TTS file too loud: {tts_file.name}")
                    
                except Exception as e:
                    self.errors.append(f"Failed to load TTS audio {tts_file.name}: {e}")
        
        logger.info(f"Found {len(tts_files)} TTS files")
    
    def validate_transcripts(self):
        """전사본 파일 검증"""
        logger.info("Validating transcript files...")
        
        transcript_dir = self.dataset_root / 'transcript'
        if not transcript_dir.exists():
            self.warnings.append("Transcript directory not found")
            return
            
        transcript_files = list(transcript_dir.glob('*.txt'))
        
        for transcript_file in transcript_files[:100]:  # 샘플링
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 기본 검증
                if not content.strip():
                    self.warnings.append(f"Empty transcript file: {transcript_file.name}")
                    continue
                    
                # 전사본 특화 검증
                lines = content.strip().split('\n')
                
                # 통계
                self.statistics['transcript_length'].append(len(content))
                self.statistics['transcript_lines'].append(len(lines))
                self.statistics['transcript_words'].append(len(content.split()))
                
                # 전사본 품질 체크
                if len(content) < 10:
                    self.warnings.append(f"Very short transcript: {transcript_file.name}")
                
                # 특수 문자나 이상한 인코딩 체크
                try:
                    content.encode('utf-8')
                except UnicodeEncodeError:
                    self.warnings.append(f"Encoding issue in transcript: {transcript_file.name}")
                
            except Exception as e:
                self.errors.append(f"Failed to read transcript file {transcript_file.name}: {e}")
        
        logger.info(f"Found {len(transcript_files)} transcript files")
    
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
        """데이터 일관성 검증 with TTS"""
        logger.info("Validating data consistency...")
        
        # 메타데이터에 있는 파일들이 실제로 존재하는지
        metadata_audio_paths = set()
        metadata_tts_paths = set()
        
        for item in self.metadata_items:
            if 'audio_path' in item:
                metadata_audio_paths.add(item['audio_path'])
            
            if item.get('tts_audio_path'):
                metadata_tts_paths.add(item['tts_audio_path'])
                
        # 실제 오디오 파일들
        audio_dir = self.dataset_root / 'audio' / 'full'
        actual_audio_files = set()
        for f in audio_dir.glob('*'):
            if f.is_file():
                rel_path = f.relative_to(self.dataset_root)
                actual_audio_files.add(str(rel_path))
        
        # 실제 TTS 파일들
        tts_dir = self.dataset_root / 'audio' / 'tts'
        actual_tts_files = set()
        if tts_dir.exists():
            for f in tts_dir.glob('*'):
                if f.is_file():
                    rel_path = f.relative_to(self.dataset_root)
                    actual_tts_files.add(str(rel_path))
                
        # 차집합 확인 - 일반 오디오
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
        
        # 차집합 확인 - TTS
        missing_tts_in_metadata = actual_tts_files - metadata_tts_paths
        missing_tts_in_filesystem = metadata_tts_paths - actual_tts_files
        
        if missing_tts_in_metadata:
            self.warnings.append(
                f"{len(missing_tts_in_metadata)} TTS files not in metadata"
            )
            
        if missing_tts_in_filesystem:
            self.warnings.append(
                f"{len(missing_tts_in_filesystem)} TTS metadata entries without files"
            )
        
        # TTS-전사본 일관성 확인
        tts_items = [item for item in self.metadata_items if item.get('has_tts', False)]
        for item in tts_items:
            if item.get('tts_audio_path') and not item.get('transcript_path'):
                self.warnings.append(
                    f"TTS audio without transcript: {item.get('id')}"
                )
            elif item.get('transcript_path') and not item.get('tts_audio_path'):
                self.warnings.append(
                    f"Transcript without TTS audio: {item.get('id')}"
                )
    
    def analyze_statistics(self):
        """통계 분석 with TTS"""
        logger.info("Analyzing dataset statistics...")
        
        # 장르 분포
        genres = []
        tts_genres = []
        
        for item in self.metadata_items:
            if 'genre' in item:
                item_genres = item['genre'] if isinstance(item['genre'], list) else [item['genre']]
                
                if item.get('is_tts_only', False) or 'tts' in item.get('audio_path', '').lower():
                    tts_genres.extend(item_genres)
                else:
                    genres.extend(item_genres)
        
        self.statistics['genre_distribution'] = Counter(genres)
        self.statistics['tts_genre_distribution'] = Counter(tts_genres)
        
        # 태스크 분포 (TTS 포함)
        task_counts = {'SONG': 0, 'INST': 0, 'COVER': 0, 'TTS': 0}
        
        for item in self.metadata_items:
            if item.get('has_tts', False) or item.get('is_tts_only', False):
                task_counts['TTS'] += 1
            elif item.get('lyric_path'):
                task_counts['SONG'] += 1
            else:
                task_counts['INST'] += 1
                
        self.statistics['task_distribution'] = task_counts
        
        # TTS vs 일반 오디오 비율
        total_items = len(self.metadata_items)
        tts_items = len([item for item in self.metadata_items if item.get('has_tts', False) or item.get('is_tts_only', False)])
        regular_items = total_items - tts_items
        
        self.statistics['data_type_distribution'] = {
            'regular': regular_items,
            'tts': tts_items,
            'total': total_items
        }
    
    def print_results(self):
        """검증 결과 출력 with TTS"""
        print("\n" + "="*60)
        print("DATASET VALIDATION RESULTS (with TTS Support)")
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
        
        # 데이터 타입 분포
        if 'data_type_distribution' in self.statistics:
            dist = self.statistics['data_type_distribution']
            print(f"\nData Type Distribution:")
            print(f"  - Regular audio: {dist['regular']}")
            print(f"  - TTS audio: {dist['tts']}")
            print(f"  - Total: {dist['total']}")
            if dist['total'] > 0:
                print(f"  - TTS ratio: {dist['tts']/dist['total']*100:.1f}%")
        
        # 오디오 통계
        if self.statistics['audio_duration']:
            durations = self.statistics['audio_duration']
            print(f"\nRegular Audio Duration:")
            print(f"  - Total: {sum(durations)/3600:.2f} hours")
            print(f"  - Average: {np.mean(durations):.2f} seconds")
            print(f"  - Min/Max: {min(durations):.2f} / {max(durations):.2f} seconds")
        
        # TTS 통계
        if self.statistics['tts_duration']:
            tts_durations = self.statistics['tts_duration']
            print(f"\nTTS Audio Duration:")
            print(f"  - Total: {sum(tts_durations)/3600:.2f} hours")
            print(f"  - Average: {np.mean(tts_durations):.2f} seconds")
            print(f"  - Min/Max: {min(tts_durations):.2f} / {max(tts_durations):.2f} seconds")
        
        # 전사본 통계
        if self.statistics['transcript_length']:
            transcript_lengths = self.statistics['transcript_length']
            transcript_words = self.statistics['transcript_words']
            print(f"\nTranscript Statistics:")
            print(f"  - Average length: {np.mean(transcript_lengths):.0f} characters")
            print(f"  - Average words: {np.mean(transcript_words):.0f}")
            print(f"  - Total transcripts: {len(transcript_lengths)}")
            
        # 장르 분포
        if self.statistics['genre_distribution']:
            print(f"\nRegular Audio Genre Distribution:")
            for genre, count in self.statistics['genre_distribution'].most_common(10):
                print(f"  - {genre}: {count}")
        
        if self.statistics['tts_genre_distribution']:
            print(f"\nTTS Genre Distribution:")
            for genre, count in self.statistics['tts_genre_distribution'].most_common(10):
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
        """통계 시각화 with TTS"""
        output_dir.mkdir(exist_ok=True)
        
        # Duration distribution comparison
        if self.statistics['audio_duration'] and self.statistics['tts_duration']:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.hist(self.statistics['audio_duration'], bins=50, alpha=0.7, label='Regular Audio')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Count')
            plt.title('Regular Audio Duration Distribution')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.hist(self.statistics['tts_duration'], bins=50, alpha=0.7, label='TTS Audio', color='orange')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Count')
            plt.title('TTS Audio Duration Distribution')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / 'duration_comparison.png')
            plt.close()
            
        # Genre distribution
        if self.statistics['genre_distribution']:
            genres = list(self.statistics['genre_distribution'].keys())[:10]
            counts = [self.statistics['genre_distribution'][g] for g in genres]
            
            plt.figure(figsize=(10, 6))
            plt.bar(genres, counts)
            plt.xlabel('Genre')
            plt.ylabel('Count')
            plt.title('Top 10 Genres (Regular Audio)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'genre_distribution.png')
            plt.close()
        
        # Task distribution pie chart
        if self.statistics['task_distribution']:
            tasks = list(self.statistics['task_distribution'].keys())
            counts = list(self.statistics['task_distribution'].values())
            
            plt.figure(figsize=(8, 8))
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            plt.pie(counts, labels=tasks, autopct='%1.1f%%', colors=colors)
            plt.title('Task Distribution')
            plt.savefig(output_dir / 'task_distribution.png')
            plt.close()
        
        # Data type distribution
        if 'data_type_distribution' in self.statistics:
            dist = self.statistics['data_type_distribution']
            
            plt.figure(figsize=(8, 6))
            types = ['Regular Audio', 'TTS Audio']
            counts = [dist['regular'], dist['tts']]
            colors = ['#66b3ff', '#ffcc99']
            
            plt.bar(types, counts, color=colors)
            plt.ylabel('Count')
            plt.title('Data Type Distribution')
            plt.savefig(output_dir / 'data_type_distribution.png')
            plt.close()


class DatasetFixer:
    """데이터셋 자동 수정 도구 with TTS support"""
    
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.fixes_applied = []
        
    def fix_broken_paths(self, metadata_file: Path):
        """깨진 경로 수정 with TTS support"""
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
                
                # TTS 경로 수정
                if item.get('tts_audio_path'):
                    tts_path = self.dataset_root / item['tts_audio_path']
                    if not tts_path.exists():
                        tts_name = Path(item['tts_audio_path']).name
                        found = list(self.dataset_root.rglob(tts_name))
                        
                        if found:
                            new_path = found[0].relative_to(self.dataset_root)
                            item['tts_audio_path'] = str(new_path)
                            self.fixes_applied.append(
                                f"Fixed TTS path: {item['id']}"
                            )
                
                # 전사본 경로 수정
                if item.get('transcript_path'):
                    transcript_path = self.dataset_root / item['transcript_path']
                    if not transcript_path.exists():
                        transcript_name = Path(item['transcript_path']).name
                        found = list(self.dataset_root.rglob(transcript_name))
                        
                        if found:
                            new_path = found[0].relative_to(self.dataset_root)
                            item['transcript_path'] = str(new_path)
                            self.fixes_applied.append(
                                f"Fixed transcript path: {item['id']}"
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
    parser = argparse.ArgumentParser(description='LYRO Dataset Validation with TTS Support')
    
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
    parser.add_argument('--check_tts', action='store_true',
                        help='Include TTS validation')
    
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
            
            for metadata_file in ['train_metadata.jsonl', 'val_metadata.jsonl', 'tts_metadata.jsonl']:
                file_path = dataset_root / 'metadata' / metadata_file
                if file_path.exists():
                    fixer.fix_broken_paths(file_path)
                    
    # 종료 코드
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()