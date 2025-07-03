#!/usr/bin/env python3
"""
LYRO 음악 생성 스크립트
훈련된 DCAE + Generator 모델을 사용한 음악 생성
"""

import os
import sys
import argparse
import torch
from pathlib import Path
import json
import time
import logging

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    create_simple_pipeline, 
    LyroGenerator, 
    GenerationConfig, 
    GenerationInput
)
from training.config import LyroConfig
from utils import safe_save_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_generation_prompts(prompt_file: str) -> list:
    """생성 프롬프트 파일 로드"""
    if not Path(prompt_file).exists():
        logger.error(f"Prompt file not found: {prompt_file}")
        return []
    
    prompts = []
    
    if prompt_file.endswith('.json'):
        # JSON 파일
        with open(prompt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                prompts = data
            elif isinstance(data, dict):
                prompts = [data]
    
    elif prompt_file.endswith('.jsonl'):
        # JSONL 파일
        with open(prompt_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    prompt = json.loads(line.strip())
                    prompts.append(prompt)
                except:
                    continue
    
    else:
        # 텍스트 파일 (각 라인이 가사)
        with open(prompt_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                lyrics = line.strip()
                if lyrics:
                    prompts.append({
                        'id': f'text_prompt_{i}',
                        'task': 'SONG',
                        'lyrics': lyrics,
                        'caption': f'A song with lyrics: {lyrics[:50]}...'
                    })
    
    logger.info(f"Loaded {len(prompts)} generation prompts")
    return prompts


def create_example_prompts() -> list:
    """예제 프롬프트 생성"""
    return [
        {
            'id': 'song_example_1',
            'task': 'SONG',
            'lyrics': '''Walking down the street tonight
The stars are shining bright
Music in my heart
A brand new start''',
            'caption': 'An upbeat pop song with electronic drums and synthesizer'
        },
        {
            'id': 'song_example_2', 
            'task': 'SONG',
            'lyrics': '''In the quiet of the morning
When the world is still asleep
I find peace in simple moments
Memories I want to keep''',
            'caption': 'A gentle acoustic ballad with guitar and soft vocals'
        },
        {
            'id': 'instrumental_example_1',
            'task': 'INST',
            'caption': 'A jazz piece featuring saxophone and piano with a walking bass line'
        },
        {
            'id': 'instrumental_example_2',
            'task': 'INST', 
            'caption': 'An electronic dance track with heavy bass and energetic synths'
        },
        {
            'id': 'instrumental_example_3',
            'task': 'INST',
            'caption': 'A classical orchestral piece with strings and brass instruments'
        }
    ]


def generate_single(
    generator: LyroGenerator,
    prompt: dict,
    config: GenerationConfig,
    output_dir: Path
) -> dict:
    """단일 프롬프트 생성"""
    start_time = time.time()
    
    try:
        # 태스크별 생성
        if prompt['task'] == 'SONG':
            result = generator.generate_song(
                lyrics=prompt['lyrics'],
                caption=prompt.get('caption'),
                duration=config.duration,
                quality=config.quality,
                seed=config.seed
            )
        
        elif prompt['task'] == 'INST':
            result = generator.generate_instrumental(
                caption=prompt['caption'],
                duration=config.duration,
                quality=config.quality,
                seed=config.seed
            )
        
        elif prompt['task'] == 'COVER':
            result = generator.generate_cover(
                lyrics=prompt['lyrics'],
                reference_audio=prompt['reference_audio'],
                caption=prompt.get('caption'),
                duration=config.duration,
                quality=config.quality,
                seed=config.seed
            )
        
        else:
            raise ValueError(f"Unknown task: {prompt['task']}")
        
        generation_time = time.time() - start_time
        
        # 파일 저장
        prompt_id = prompt.get('id', 'unknown')
        output_path = output_dir / f"{prompt_id}.wav"
        
        success = safe_save_audio(
            audio=result['audio'],
            path=output_path,
            sample_rate=result['sample_rate']
        )
        
        if success:
            # 메타데이터 저장
            metadata = {
                'prompt': prompt,
                'config': config.__dict__,
                'generation_time': generation_time,
                'model_info': result.get('metadata', {}),
                'audio_info': {
                    'sample_rate': result['sample_rate'],
                    'duration': result['duration'],
                    'channels': result['audio'].shape[0] if hasattr(result['audio'], 'shape') else 2
                }
            }
            
            metadata_path = output_dir / f"{prompt_id}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Generated {prompt_id} in {generation_time:.2f}s -> {output_path}")
            
            return {
                'success': True,
                'prompt_id': prompt_id,
                'output_path': str(output_path),
                'generation_time': generation_time,
                'result': result
            }
        else:
            logger.error(f"❌ Failed to save {prompt_id}")
            return {'success': False, 'prompt_id': prompt_id, 'error': 'Save failed'}
    
    except Exception as e:
        logger.error(f"❌ Generation failed for {prompt.get('id', 'unknown')}: {e}")
        return {'success': False, 'prompt_id': prompt.get('id', 'unknown'), 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='LYRO Music Generation')
    
    # 모델 체크포인트
    parser.add_argument('--dcae_checkpoint', type=str, required=True, help='DCAE checkpoint path')
    parser.add_argument('--generator_checkpoint', type=str, required=True, help='Generator checkpoint path')
    
    # 생성 설정
    parser.add_argument('--prompts', type=str, default=None, help='Prompts file (JSON/JSONL/TXT)')
    parser.add_argument('--output_dir', type=str, default='generated_music', help='Output directory')
    parser.add_argument('--duration', type=float, default=10.0, help='Generation duration (seconds)')
    parser.add_argument('--quality', type=str, default='standard', choices=['fast', 'standard', 'high'], help='Generation quality')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    # 단일 생성 옵션
    parser.add_argument('--lyrics', type=str, default=None, help='Single lyrics input')
    parser.add_argument('--caption', type=str, default=None, help='Single caption input')
    parser.add_argument('--task', type=str, default='SONG', choices=['SONG', 'INST', 'COVER'], help='Generation task')
    
    # 기타 설정
    parser.add_argument('--batch_size', type=int, default=1, help='Batch generation size')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파이프라인 생성
    logger.info("Loading models...")
    try:
        pipeline = create_simple_pipeline(
            dcae_checkpoint=args.dcae_checkpoint,
            generator_checkpoint=args.generator_checkpoint,
            device=device
        )
        
        generator = LyroGenerator(pipeline)
        logger.info("✅ Models loaded successfully")
        
        # 모델 정보 출력
        model_info = pipeline.get_model_info()
        logger.info(f"Model info: {model_info}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        return 1
    
    # 생성 설정
    generation_config = GenerationConfig(
        duration=args.duration,
        quality=args.quality,
        seed=args.seed
    )
    
    # 프롬프트 준비
    if args.prompts:
        # 파일에서 프롬프트 로드
        prompts = load_generation_prompts(args.prompts)
        if not prompts:
            logger.error("No valid prompts found")
            return 1
    
    elif args.lyrics or args.caption:
        # 단일 프롬프트
        prompt = {
            'id': 'single_generation',
            'task': args.task,
        }
        
        if args.lyrics:
            prompt['lyrics'] = args.lyrics
        if args.caption:
            prompt['caption'] = args.caption
        
        prompts = [prompt]
    
    else:
        # 예제 프롬프트 사용
        logger.info("No prompts provided, using examples")
        prompts = create_example_prompts()
    
    logger.info(f"Generating {len(prompts)} samples...")
    
    # 생성 실행
    results = []
    total_start_time = time.time()
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Generating {i+1}/{len(prompts)}: {prompt.get('id', 'unknown')}")
        
        result = generate_single(
            generator=generator,
            prompt=prompt,
            config=generation_config,
            output_dir=output_dir
        )
        
        results.append(result)
    
    total_time = time.time() - total_start_time
    
    # 결과 요약
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    logger.info(f"\n🎵 Generation Summary:")
    logger.info(f"   ✅ Successful: {len(successful)}")
    logger.info(f"   ❌ Failed: {len(failed)}")
    logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
    logger.info(f"   📁 Output directory: {output_dir}")
    
    if successful:
        avg_time = sum(r['generation_time'] for r in successful) / len(successful)
        logger.info(f"   📊 Average generation time: {avg_time:.2f}s")
    
    # 실패한 생성 로그
    if failed:
        logger.info("\n❌ Failed generations:")
        for result in failed:
            logger.info(f"   - {result['prompt_id']}: {result.get('error', 'Unknown error')}")
    
    # 결과 요약 저장
    summary = {
        'total_prompts': len(prompts),
        'successful': len(successful),
        'failed': len(failed),
        'total_time': total_time,
        'average_time': avg_time if successful else 0,
        'config': generation_config.__dict__,
        'results': results
    }
    
    summary_path = output_dir / 'generation_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"📋 Summary saved to {summary_path}")
    
    return 0 if not failed else 1


if __name__ == '__main__':
    exit(main())
