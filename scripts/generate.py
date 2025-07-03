#!/usr/bin/env python3
"""
LYRO 음악 생성 스크립트 (Vocoder 통합 버전)
훈련된 DCAE + Vocoder + Generator 모델을 사용한 고품질 음악 생성
"""

import os
import sys
import argparse
import torch
from pathlib import Path
import json
import time
import logging
from typing import Dict, List, Any

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    create_pipeline, 
    LyroGenerator, 
    GenerationConfig, 
    GenerationInput,
    quick_generate
)
from training.config import LyroConfig
from utils import safe_save_audio, MetricCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VocoderComparison:
    """Vocoder vs DCAE 품질 비교"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.metric_calculator = MetricCalculator(sample_rate)
    
    def compare_quality(
        self, 
        pipeline, 
        input_data: GenerationInput,
        duration: float = 10.0
    ) -> Dict[str, Any]:
        """Vocoder와 DCAE-only 비교"""
        
        results = {}
        
        try:
            # Vocoder 사용 생성
            logger.info("🎤 Generating with Vocoder...")
            vocoder_config = GenerationConfig(
                duration=duration,
                quality="standard",
                use_vocoder=True,
                seed=42  # 동일한 시드로 공정한 비교
            )
            
            vocoder_result = pipeline.generate(input_data, vocoder_config, verbose=False)
            results['vocoder'] = {
                'audio': vocoder_result['audio'],
                'generation_time': vocoder_result['generation_time'],
                'quality_metrics': vocoder_result.get('quality_metrics', {}),
                'config': vocoder_config
            }
            
            # DCAE-only 생성
            logger.info("🔄 Generating with DCAE only...")
            dcae_config = GenerationConfig(
                duration=duration,
                quality="standard", 
                use_vocoder=False,
                seed=42  # 동일한 시드
            )
            
            dcae_result = pipeline.generate(input_data, dcae_config, verbose=False)
            results['dcae_only'] = {
                'audio': dcae_result['audio'],
                'generation_time': dcae_result['generation_time'],
                'quality_metrics': dcae_result.get('quality_metrics', {}),
                'config': dcae_config
            }
            
            # 품질 비교 메트릭
            logger.info("📊 Comparing quality metrics...")
            comparison_metrics = self._compute_comparison_metrics(
                vocoder_audio=results['vocoder']['audio'],
                dcae_audio=results['dcae_only']['audio']
            )
            results['comparison'] = comparison_metrics
            
            # 요약
            results['summary'] = self._generate_comparison_summary(results)
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _compute_comparison_metrics(
        self,
        vocoder_audio: torch.Tensor,
        dcae_audio: torch.Tensor
    ) -> Dict[str, float]:
        """두 오디오 간 비교 메트릭 계산"""
        
        try:
            # 상호 메트릭 계산
            metrics = self.metric_calculator.compute_all_metrics(
                target_audio=vocoder_audio,
                generated_audio=dcae_audio
            )
            
            comparison = {}
            
            # 개별 품질 점수
            vocoder_quality = self._calculate_audio_quality(vocoder_audio)
            dcae_quality = self._calculate_audio_quality(dcae_audio)
            
            comparison['vocoder_quality'] = vocoder_quality
            comparison['dcae_quality'] = dcae_quality
            comparison['quality_improvement'] = vocoder_quality - dcae_quality
            
            # 스펙트럴 유사도
            if 'spectral_convergence' in metrics:
                comparison['spectral_similarity'] = 1.0 - metrics['spectral_convergence'].value
            
            # 멜 거리
            if 'mel_distance' in metrics:
                comparison['mel_distance'] = metrics['mel_distance'].value
            
            # 하모닉 유사도
            if 'harmonic_similarity' in metrics:
                comparison['harmonic_similarity'] = metrics['harmonic_similarity'].value
            
            return comparison
            
        except Exception as e:
            logger.warning(f"Comparison metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_audio_quality(self, audio: torch.Tensor) -> float:
        """단일 오디오의 품질 점수 계산"""
        try:
            # RMS 에너지
            rms = torch.sqrt(torch.mean(audio ** 2)).item()
            
            # 다이나믹 레인지
            peak = torch.max(torch.abs(audio)).item()
            dynamic_range = peak / (rms + 1e-10)
            
            # 클리핑 비율
            clipping_ratio = torch.mean((torch.abs(audio) >= 0.99).float()).item()
            
            # 무음 비율
            silence_threshold = 0.01
            silence_ratio = torch.mean((torch.abs(audio) < silence_threshold).float()).item()
            
            # 종합 품질 점수
            quality_score = (
                min(rms * 10, 1.0) * 0.3 +
                min(dynamic_range / 10, 1.0) * 0.3 +
                (1.0 - clipping_ratio) * 0.2 +
                (1.0 - min(silence_ratio, 1.0)) * 0.2
            )
            
            return quality_score
            
        except Exception:
            return 0.5
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """비교 결과 요약 생성"""
        summary = {}
        
        try:
            vocoder_data = results.get('vocoder', {})
            dcae_data = results.get('dcae_only', {})
            comparison = results.get('comparison', {})
            
            # 생성 시간 비교
            vocoder_time = vocoder_data.get('generation_time', 0)
            dcae_time = dcae_data.get('generation_time', 0)
            time_overhead = vocoder_time - dcae_time
            
            summary['generation_time'] = {
                'vocoder': vocoder_time,
                'dcae_only': dcae_time,
                'overhead': time_overhead,
                'overhead_percentage': (time_overhead / max(dcae_time, 0.1)) * 100
            }
            
            # 품질 비교
            quality_improvement = comparison.get('quality_improvement', 0)
            
            summary['quality'] = {
                'vocoder_score': comparison.get('vocoder_quality', 0),
                'dcae_score': comparison.get('dcae_quality', 0),
                'improvement': quality_improvement,
                'improvement_percentage': quality_improvement * 100,
                'vocoder_better': quality_improvement > 0.05  # 5% 이상 개선
            }
            
            # 추천
            if quality_improvement > 0.1 and time_overhead < 5.0:
                recommendation = "Use Vocoder (significant quality improvement with acceptable overhead)"
            elif quality_improvement > 0.05:
                recommendation = "Use Vocoder (noticeable quality improvement)"
            elif time_overhead > 10.0:
                recommendation = "Use DCAE-only (Vocoder overhead too high)"
            else:
                recommendation = "Use DCAE-only (minimal quality difference)"
            
            summary['recommendation'] = recommendation
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary


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
    """예제 프롬프트 생성 (Vocoder 테스트용)"""
    return [
        {
            'id': 'vocoder_test_song_1',
            'task': 'SONG',
            'lyrics': '''Walking down the street tonight
The stars are shining bright
Music in my heart
A brand new start
Every step I take
Every move I make''',
            'caption': 'An upbeat pop song with electronic drums and synthesizer'
        },
        {
            'id': 'vocoder_test_song_2', 
            'task': 'SONG',
            'lyrics': '''In the quiet of the morning
When the world is still asleep
I find peace in simple moments
Memories I want to keep
Soft whispers of the wind
Gentle touch of sunlight''',
            'caption': 'A gentle acoustic ballad with guitar and soft vocals'
        },
        {
            'id': 'vocoder_test_instrumental_1',
            'task': 'INST',
            'caption': 'A complex jazz piece featuring saxophone, piano, and walking bass line with intricate harmonies'
        },
        {
            'id': 'vocoder_test_instrumental_2',
            'task': 'INST', 
            'caption': 'An energetic electronic dance track with heavy bass, dynamic synths, and pulsing rhythm'
        },
        {
            'id': 'vocoder_test_classical',
            'task': 'INST',
            'caption': 'A majestic classical orchestral piece with strings, brass, and woodwinds featuring rich harmonies'
        }
    ]


def generate_single(
    generator: LyroGenerator,
    prompt: dict,
    config: GenerationConfig,
    output_dir: Path,
    save_comparison: bool = False
) -> dict:
    """단일 프롬프트 생성 (Vocoder 옵션 포함)"""
    start_time = time.time()
    
    try:
        # 생성 입력 준비
        generation_input = GenerationInput(
            task=prompt['task'],
            lyrics=prompt.get('lyrics'),
            caption=prompt.get('caption'),
            reference_audio=prompt.get('reference_audio'),
            genre=prompt.get('genre'),
            mood=prompt.get('mood'),
            tempo=prompt.get('tempo')
        )
        
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
        
        # Vocoder 사용 여부에 따른 파일명 구분
        vocoder_suffix = "_vocoder" if config.use_vocoder else "_dcae"
        output_path = output_dir / f"{prompt_id}{vocoder_suffix}.wav"
        
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
                },
                'vocoder_used': config.use_vocoder,
                'quality_metrics': result.get('quality_metrics', {})
            }
            
            metadata_path = output_dir / f"{prompt_id}{vocoder_suffix}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            # 품질 비교 수행 (요청된 경우)
            comparison_result = None
            if save_comparison and hasattr(generator, 'pipeline'):
                try:
                    comparator = VocoderComparison()
                    comparison_result = comparator.compare_quality(
                        generator.pipeline, 
                        generation_input,
                        config.duration
                    )
                    
                    # 비교 결과 저장
                    comparison_path = output_dir / f"{prompt_id}_comparison.json"
                    with open(comparison_path, 'w', encoding='utf-8') as f:
                        json.dump(comparison_result, f, indent=2, ensure_ascii=False, default=str)
                    
                    # 비교 오디오도 저장
                    if 'dcae_only' in comparison_result:
                        dcae_audio_path = output_dir / f"{prompt_id}_dcae_comparison.wav"
                        safe_save_audio(
                            audio=comparison_result['dcae_only']['audio'],
                            path=dcae_audio_path,
                            sample_rate=result['sample_rate']
                        )
                    
                except Exception as e:
                    logger.warning(f"Quality comparison failed for {prompt_id}: {e}")
            
            vocoder_info = "with Vocoder" if config.use_vocoder else "DCAE-only"
            logger.info(f"✅ Generated {prompt_id} ({vocoder_info}) in {generation_time:.2f}s -> {output_path}")
            
            generation_result = {
                'success': True,
                'prompt_id': prompt_id,
                'output_path': str(output_path),
                'generation_time': generation_time,
                'vocoder_used': config.use_vocoder,
                'quality_score': result.get('quality_metrics', {}).get('overall_quality', 'N/A'),
                'result': result
            }
            
            if comparison_result:
                generation_result['comparison'] = comparison_result
            
            return generation_result
            
        else:
            logger.error(f"❌ Failed to save {prompt_id}")
            return {'success': False, 'prompt_id': prompt_id, 'error': 'Save failed'}
    
    except Exception as e:
        logger.error(f"❌ Generation failed for {prompt.get('id', 'unknown')}: {e}")
        return {'success': False, 'prompt_id': prompt.get('id', 'unknown'), 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='LYRO Music Generation (with Vocoder support)')
    
    # 모델 체크포인트
    parser.add_argument('--dcae_checkpoint', type=str, required=True, help='DCAE checkpoint path')
    parser.add_argument('--generator_checkpoint', type=str, required=True, help='Generator checkpoint path')
    
    # 생성 설정
    parser.add_argument('--prompts', type=str, default=None, help='Prompts file (JSON/JSONL/TXT)')
    parser.add_argument('--output_dir', type=str, default='generated_music', help='Output directory')
    parser.add_argument('--duration', type=float, default=10.0, help='Generation duration (seconds)')
    parser.add_argument('--quality', type=str, default='standard', choices=['fast', 'standard', 'high'], help='Generation quality')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    # Vocoder 설정
    parser.add_argument('--use_vocoder', action='store_true', default=True, help='Use Vocoder for high-quality synthesis')
    parser.add_argument('--no_vocoder', action='store_true', help='Force disable Vocoder (use DCAE only)')
    parser.add_argument('--vocoder_quality', type=str, default='standard', choices=['fast', 'standard', 'high'], help='Vocoder quality level')
    parser.add_argument('--compare_quality', action='store_true', help='Generate both Vocoder and DCAE versions for comparison')
    parser.add_argument('--comparison_only', action='store_true', help='Only perform quality comparison without saving individual files')
    
    # 단일 생성 옵션
    parser.add_argument('--lyrics', type=str, default=None, help='Single lyrics input')
    parser.add_argument('--caption', type=str, default=None, help='Single caption input')
    parser.add_argument('--task', type=str, default='SONG', choices=['SONG', 'INST', 'COVER'], help='Generation task')
    
    # 기타 설정
    parser.add_argument('--batch_size', type=int, default=1, help='Batch generation size')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # Vocoder 사용 결정
    use_vocoder = args.use_vocoder and not args.no_vocoder
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Vocoder enabled: {use_vocoder}")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파이프라인 생성
    logger.info("Loading models...")
    try:
        pipeline = create_pipeline(
            dcae_checkpoint=args.dcae_checkpoint,
            generator_checkpoint=args.generator_checkpoint,
            use_vocoder=use_vocoder,
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
        use_vocoder=use_vocoder,
        vocoder_quality=args.vocoder_quality,
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
        # 예제 프롬프트 사용 (Vocoder 테스트용)
        logger.info("No prompts provided, using Vocoder test examples")
        prompts = create_example_prompts()
    
    logger.info(f"Generating {len(prompts)} samples...")
    
    # 품질 비교 모드
    if args.comparison_only or args.compare_quality:
        logger.info("🔬 Quality comparison mode enabled")
        comparator = VocoderComparison()
        
        comparison_results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Comparing {i+1}/{len(prompts)}: {prompt.get('id', 'unknown')}")
            
            generation_input = GenerationInput(
                task=prompt['task'],
                lyrics=prompt.get('lyrics'),
                caption=prompt.get('caption'),
                reference_audio=prompt.get('reference_audio')
            )
            
            comparison_result = comparator.compare_quality(
                pipeline, 
                generation_input,
                args.duration
            )
            
            # 결과 저장
            prompt_id = prompt.get('id', f'comparison_{i}')
            comparison_path = output_dir / f"{prompt_id}_quality_comparison.json"
            
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False, default=str)
            
            # 비교 오디오 저장
            if 'vocoder' in comparison_result:
                vocoder_path = output_dir / f"{prompt_id}_vocoder.wav"
                safe_save_audio(
                    audio=comparison_result['vocoder']['audio'],
                    path=vocoder_path,
                    sample_rate=44100
                )
            
            if 'dcae_only' in comparison_result:
                dcae_path = output_dir / f"{prompt_id}_dcae.wav"
                safe_save_audio(
                    audio=comparison_result['dcae_only']['audio'],
                    path=dcae_path,
                    sample_rate=44100
                )
            
            comparison_results.append(comparison_result)
            
            # 비교 요약 출력
            if 'summary' in comparison_result:
                summary = comparison_result['summary']
                logger.info(f"   📊 {prompt_id} comparison:")
                logger.info(f"      Quality improvement: {summary.get('quality', {}).get('improvement_percentage', 0):.1f}%")
                logger.info(f"      Time overhead: {summary.get('generation_time', {}).get('overhead', 0):.2f}s")
                logger.info(f"      Recommendation: {summary.get('recommendation', 'N/A')}")
        
        # 전체 비교 요약
        summary_path = output_dir / 'overall_comparison_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_comparisons': len(comparison_results),
                'results': comparison_results
            }, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"🎯 Quality comparison completed! Results saved to {output_dir}")
        
        return 0
    
    # 일반 생성 모드
    results = []
    total_start_time = time.time()
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Generating {i+1}/{len(prompts)}: {prompt.get('id', 'unknown')}")
        
        result = generate_single(
            generator=generator,
            prompt=prompt,
            config=generation_config,
            output_dir=output_dir,
            save_comparison=args.compare_quality
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
    logger.info(f"   🎤 Vocoder used: {use_vocoder}")
    logger.info(f"   📁 Output directory: {output_dir}")
    
    if successful:
        avg_time = sum(r['generation_time'] for r in successful) / len(successful)
        avg_quality = sum(
            r.get('quality_score', 0) for r in successful 
            if isinstance(r.get('quality_score'), (int, float))
        ) / max(len([r for r in successful if isinstance(r.get('quality_score'), (int, float))]), 1)
        
        logger.info(f"   📊 Average generation time: {avg_time:.2f}s")
        logger.info(f"   🎯 Average quality score: {avg_quality:.3f}")
    
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
        'average_quality': avg_quality if successful else 0,
        'config': generation_config.__dict__,
        'vocoder_enabled': use_vocoder,
        'results': results
    }
    
    summary_path = output_dir / 'generation_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"📋 Summary saved to {summary_path}")
    
    return 0 if not failed else 1


if __name__ == '__main__':
    exit(main())