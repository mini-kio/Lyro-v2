#!/usr/bin/env python3
"""
Vocabulary Builder for UnifiedTextEncoder
데이터셋에서 동적으로 vocab 구축
"""

import json
import os
from typing import List, Dict, Set, Tuple
from collections import Counter
import re

class VocabBuilder:
    """동적 vocab 구축 도구"""
    
    def __init__(self):
        self.section_counter = Counter()
        self.instrument_counter = Counter()
        self.tag_patterns = []
        
    def parse_structured_text(self, text: str) -> Tuple[str, str, str]:
        """구조화된 텍스트 파싱"""
        # 다양한 패턴 지원
        patterns = [
            r'\[([^:]+):([^]]+)\]\s*(.*)',  # [section:instrument] content
            r'\[([^]]+)\]\s*(.*)',          # [section] content (악기 없음)
            r'(\w+)\s*:\s*([^,]+),?\s*(.*)',  # section: instrument, content
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text.strip())
            if match:
                if len(match.groups()) == 3:
                    section = match.group(1).strip().lower()
                    instrument = match.group(2).strip().lower()
                    content = match.group(3).strip()
                    return content, section, instrument
                elif len(match.groups()) == 2:
                    section = match.group(1).strip().lower()
                    content = match.group(2).strip()
                    return content, section, ""
        
        return text, "", ""
    
    def analyze_dataset(self, texts: List[str]) -> Dict:
        """데이터셋 분석 및 vocab 구축"""
        total_texts = len(texts)
        structured_count = 0
        
        for text in texts:
            content, section, instrument = self.parse_structured_text(text)
            
            if section or instrument:
                structured_count += 1
                
            if section:
                self.section_counter[section] += 1
            if instrument:
                self.instrument_counter[instrument] += 1
        
        # 통계 계산
        stats = {
            'total_texts': total_texts,
            'structured_texts': structured_count,
            'structured_ratio': structured_count / total_texts if total_texts > 0 else 0,
            'unique_sections': len(self.section_counter),
            'unique_instruments': len(self.instrument_counter),
            'section_distribution': dict(self.section_counter.most_common()),
            'instrument_distribution': dict(self.instrument_counter.most_common())
        }
        
        return stats
    
    def build_vocab(self, min_frequency: int = 2) -> Tuple[Dict[str, int], Dict[str, int]]:
        """최소 빈도 기반으로 vocab 구축"""
        
        # 빈도 필터링
        filtered_sections = {
            section: count for section, count in self.section_counter.items()
            if count >= min_frequency
        }
        
        filtered_instruments = {
            instrument: count for instrument, count in self.instrument_counter.items()
            if count >= min_frequency
        }
        
        # vocab 생성 (빈도순)
        section_vocab = {}
        for i, (section, _) in enumerate(
            sorted(filtered_sections.items(), key=lambda x: x[1], reverse=True)
        ):
            section_vocab[section] = i
        
        instrument_vocab = {}
        for i, (instrument, _) in enumerate(
            sorted(filtered_instruments.items(), key=lambda x: x[1], reverse=True)
        ):
            instrument_vocab[instrument] = i
        
        return section_vocab, instrument_vocab
    
    def save_analysis(self, stats: Dict, vocab_path: str):
        """분석 결과 저장"""
        section_vocab, instrument_vocab = self.build_vocab()
        
        analysis_data = {
            'stats': stats,
            'section_vocab': section_vocab,
            'instrument_vocab': instrument_vocab,
            'section_frequencies': dict(self.section_counter),
            'instrument_frequencies': dict(self.instrument_counter)
        }
        
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
        print(f"Analysis saved to {vocab_path}")
        return analysis_data
    
    def load_analysis(self, vocab_path: str) -> Dict:
        """분석 결과 로드"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def analyze_dataset_file(dataset_path: str, text_field: str = 'text') -> Dict:
    """데이터셋 파일 분석"""
    builder = VocabBuilder()
    texts = []
    
    # JSONL 파일 읽기
    if dataset_path.endswith('.jsonl'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if text_field in data:
                    texts.append(data[text_field])
    
    # JSON 파일 읽기
    elif dataset_path.endswith('.json'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if text_field in item:
                        texts.append(item[text_field])
    
    stats = builder.analyze_dataset(texts)
    
    # 결과 출력
    print(f"\n=== Dataset Analysis: {dataset_path} ===")
    print(f"Total texts: {stats['total_texts']:,}")
    print(f"Structured texts: {stats['structured_texts']:,} ({stats['structured_ratio']:.1%})")
    print(f"Unique sections: {stats['unique_sections']}")
    print(f"Unique instruments: {stats['unique_instruments']}")
    
    print(f"\nTop sections:")
    for section, count in list(stats['section_distribution'].items())[:10]:
        print(f"  {section}: {count}")
    
    print(f"\nTop instruments:")
    for instrument, count in list(stats['instrument_distribution'].items())[:10]:
        print(f"  {instrument}: {count}")
    
    # vocab 저장
    vocab_path = dataset_path.replace('.jsonl', '_vocab.json').replace('.json', '_vocab.json')
    builder.save_analysis(stats, vocab_path)
    
    return stats


def create_unified_vocab(vocab_files: List[str], output_path: str):
    """여러 vocab 파일을 통합"""
    unified_section_counter = Counter()
    unified_instrument_counter = Counter()
    
    for vocab_file in vocab_files:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 빈도 누적
        section_freq = data.get('section_frequencies', {})
        instrument_freq = data.get('instrument_frequencies', {})
        
        unified_section_counter.update(section_freq)
        unified_instrument_counter.update(instrument_freq)
    
    # 통합 vocab 생성
    section_vocab = {}
    for i, (section, _) in enumerate(unified_section_counter.most_common()):
        section_vocab[section] = i
    
    instrument_vocab = {}
    for i, (instrument, _) in enumerate(unified_instrument_counter.most_common()):
        instrument_vocab[instrument] = i
    
    unified_data = {
        'section_vocab': section_vocab,
        'instrument_vocab': instrument_vocab,
        'section_frequencies': dict(unified_section_counter),
        'instrument_frequencies': dict(unified_instrument_counter),
        'total_sections': len(section_vocab),
        'total_instruments': len(instrument_vocab)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, ensure_ascii=False, indent=2)
    
    print(f"Unified vocab saved to {output_path}")
    print(f"Total sections: {len(section_vocab)}")
    print(f"Total instruments: {len(instrument_vocab)}")
    
    return unified_data


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vocabulary.py <dataset_path> [text_field]")
        print("Example: python vocabulary.py dataset/train.jsonl lyrics")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    text_field = sys.argv[2] if len(sys.argv) > 2 else 'text'
    
    analyze_dataset_file(dataset_path, text_field)