# lyro/inference/generator.py
"""
LYRO 생성기 클래스들 (수정됨 - Generator 전용)
Latent Vector 생성을 위한 다양한 생성 시나리오
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time
import logging

from .pipeline import LyroPipeline, GenerationConfig, GenerationInput

logger = logging.getLogger(__name__)


class LyroGenerator:
    """
    LYRO 메인 생성기 (수정됨 - Latent 출력)
    단일 생성부터 배치 생성까지 지원
    """
    
    def __init__(self, pipeline: LyroPipeline):
        self.pipeline = pipeline
        self.generation_history: List[Dict] = []
    
    def generate_song(
        self,
        lyrics: str,
        caption: Optional[str] = None,
        duration: float = 10.0,
        quality: str = "standard",
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        가사 기반 노래 latent 생성
        
        Args:
            lyrics: 가사 텍스트
            caption: 추가 캡션 (선택적)
            duration: 생성 시간 (초)
            quality: 품질 설정 (fast, standard, high)
            seed: 랜덤 시드
            
        Returns:
            생성 결과 (latent vectors)
        """
        input_data = GenerationInput(
            task="SONG",
            lyrics=lyrics,
            caption=caption
        )
        
        config = GenerationConfig(
            duration=duration,
            quality=quality,
            seed=seed
        )
        
        result = self.pipeline.generate(input_data, config)
        
        # 히스토리에 추가
        self.generation_history.append({
            'type': 'song',
            'input': input_data,
            'config': config,
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def generate_instrumental(
        self,
        caption: str,
        duration: float = 10.0,
        quality: str = "standard",
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        캡션 기반 인스트루멘탈 latent 생성
        
        Args:
            caption: MusicCaps 스타일 캡션
            duration: 생성 시간 (초)
            quality: 품질 설정
            seed: 랜덤 시드
            
        Returns:
            생성 결과 (latent vectors)
        """
        input_data = GenerationInput(
            task="INST",
            caption=caption
        )
        
        config = GenerationConfig(
            duration=duration,
            quality=quality,
            seed=seed
        )
        
        result = self.pipeline.generate(input_data, config)
        
        # 히스토리에 추가
        self.generation_history.append({
            'type': 'instrumental',
            'input': input_data,
            'config': config,
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def generate_cover(
        self,
        lyrics: str,
        reference_latents: Union[torch.Tensor, np.ndarray],
        caption: Optional[str] = None,
        duration: float = 10.0,
        quality: str = "standard",
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        참조 latent 기반 커버 생성
        
        Args:
            lyrics: 가사 텍스트
            reference_latents: 참조 latent vectors
            caption: 추가 캡션
            duration: 생성 시간 (초)
            quality: 품질 설정
            seed: 랜덤 시드
            
        Returns:
            생성 결과 (latent vectors)
        """
        input_data = GenerationInput(
            task="COVER",
            lyrics=lyrics,
            reference_latents=reference_latents,
            caption=caption
        )
        
        config = GenerationConfig(
            duration=duration,
            quality=quality,
            seed=seed
        )
        
        result = self.pipeline.generate(input_data, config)
        
        # 히스토리에 추가
        self.generation_history.append({
            'type': 'cover',
            'input': input_data,
            'config': config,
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def get_history(self) -> List[Dict]:
        """생성 히스토리 반환"""
        return self.generation_history.copy()
    
    def clear_history(self):
        """생성 히스토리 초기화"""
        self.generation_history.clear()


class SampleGenerator:
    """
    샘플링 기반 생성기 (수정됨 - Latent 출력)
    다양한 샘플링 전략 지원
    """
    
    def __init__(self, pipeline: LyroPipeline):
        self.pipeline = pipeline
    
    def sample_with_temperature(
        self,
        input_data: GenerationInput,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        온도 기반 샘플링
        
        Args:
            input_data: 생성 입력
            temperature: 샘플링 온도
            top_k: Top-k 샘플링 (선택적)
            top_p: Top-p (nucleus) 샘플링 (선택적)
            **kwargs: 추가 생성 설정
            
        Returns:
            생성 결과 (latent vectors)
        """
        config = GenerationConfig(**kwargs)
        
        # 온도 설정을 Flow Matching에 적용
        config.cfg_scale = config.cfg_scale / temperature
        
        return self.pipeline.generate(input_data, config)
    
    def iterative_refinement(
        self,
        input_data: GenerationInput,
        num_iterations: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        반복적 개선 생성
        
        Args:
            input_data: 생성 입력
            num_iterations: 반복 횟수
            **kwargs: 추가 생성 설정
            
        Returns:
            반복별 생성 결과 리스트 (latent vectors)
        """
        results = []
        
        for i in range(num_iterations):
            config = GenerationConfig(**kwargs)
            
            # 반복마다 품질 향상
            if i == 0:
                config.quality = "fast"
            elif i == num_iterations - 1:
                config.quality = "high"
            else:
                config.quality = "standard"
            
            # 이전 결과를 참조로 사용 (두 번째 반복부터)
            if i > 0 and results:
                prev_latents = results[-1]['latents']
                input_data.reference_latents = prev_latents
                input_data.task = "COVER"
            
            result = self.pipeline.generate(input_data, config)
            results.append(result)
        
        return results


class BatchGenerator:
    """
    배치 생성기 (수정됨 - Latent 출력)
    대량 latent 생성을 위한 효율적인 처리
    """
    
    def __init__(self, pipeline: LyroPipeline):
        self.pipeline = pipeline
    
    def generate_batch(
        self,
        inputs: List[GenerationInput],
        configs: Optional[List[GenerationConfig]] = None,
        max_parallel: int = 4
    ) -> List[Dict[str, Any]]:
        """
        배치 생성
        
        Args:
            inputs: 생성 입력 리스트
            configs: 생성 설정 리스트 (선택적)
            max_parallel: 최대 병렬 처리 수
            
        Returns:
            생성 결과 리스트 (latent vectors)
        """
        if configs is None:
            configs = [GenerationConfig() for _ in inputs]
        
        if len(configs) != len(inputs):
            raise ValueError("configs length must match inputs length")
        
        results = []
        
        # 청크 단위로 처리
        for i in range(0, len(inputs), max_parallel):
            chunk_inputs = inputs[i:i + max_parallel]
            chunk_configs = configs[i:i + max_parallel]
            
            # 청크 내 병렬 처리
            chunk_results = self._process_chunk(chunk_inputs, chunk_configs)
            results.extend(chunk_results)
        
        return results
    
    def _process_chunk(
        self,
        inputs: List[GenerationInput],
        configs: List[GenerationConfig]
    ) -> List[Dict[str, Any]]:
        """청크 내 병렬 처리"""
        import concurrent.futures
        
        def generate_single(input_config_pair):
            input_data, config = input_config_pair
            try:
                return self.pipeline.generate(input_data, config)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return {'error': str(e)}
        
        # 병렬 처리
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(generate_single, (inp, cfg))
                for inp, cfg in zip(inputs, configs)
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Future result failed: {e}")
                    results.append({'error': str(e)})
        
        return results
    
    def generate_variations(
        self,
        base_input: GenerationInput,
        num_variations: int = 5,
        variation_strength: float = 0.5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        변형 생성
        
        Args:
            base_input: 기본 입력
            num_variations: 변형 개수
            variation_strength: 변형 강도 (0.0-1.0)
            **kwargs: 추가 생성 설정
            
        Returns:
            변형 결과 리스트 (latent vectors)
        """
        results = []
        
        for i in range(num_variations):
            # 각 변형마다 다른 시드 사용
            config = GenerationConfig(
                seed=42 + i,
                **kwargs
            )
            
            # 변형 강도에 따라 CFG 스케일 조정
            config.cfg_scale = config.cfg_scale * (1.0 + variation_strength * (i / num_variations - 0.5))
            
            result = self.pipeline.generate(base_input, config)
            results.append(result)
        
        return results
    
    def generate_interpolations(
        self,
        input_a: GenerationInput,
        input_b: GenerationInput,
        num_steps: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        두 입력 간 보간 생성 (수정됨 - Latent 기반)
        
        Args:
            input_a: 첫 번째 입력
            input_b: 두 번째 입력  
            num_steps: 보간 스텝 수
            **kwargs: 추가 생성 설정
            
        Returns:
            보간 결과 리스트 (latent vectors)
        """
        return self.pipeline.interpolate_latents(
            input_a=input_a,
            input_b=input_b,
            num_steps=num_steps,
            generation_config=GenerationConfig(**kwargs)
        )
    
    def generate_dataset(
        self,
        output_dir: Union[str, Path],
        num_samples: int = 1000,
        tasks: List[str] = None,
        save_interval: int = 100
    ) -> List[str]:
        """
        대규모 latent 데이터셋 생성
        
        Args:
            output_dir: 출력 디렉토리
            num_samples: 생성할 샘플 수
            tasks: 생성할 태스크 리스트
            save_interval: 저장 간격
            
        Returns:
            생성된 파일 경로 리스트
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if tasks is None:
            tasks = ['SONG', 'INST', 'COVER']
        
        generated_files = []
        
        for i in range(num_samples):
            # 랜덤 태스크 선택
            task = np.random.choice(tasks)
            
            # 입력 생성
            if task == 'SONG':
                input_data = GenerationInput(
                    task=task,
                    lyrics=f"Sample lyrics for song {i}",
                    caption=f"A music piece #{i}"
                )
            elif task == 'INST':
                input_data = GenerationInput(
                    task=task,
                    caption=f"Instrumental music piece #{i}"
                )
            else:  # COVER
                # 이전 생성된 latent를 참조로 사용
                if generated_files:
                    prev_latents = self.pipeline.load_latents(generated_files[-1])
                    input_data = GenerationInput(
                        task=task,
                        lyrics=f"Cover lyrics for song {i}",
                        reference_latents=prev_latents
                    )
                else:
                    # 첫 번째이면 SONG으로 변경
                    input_data = GenerationInput(
                        task='SONG',
                        lyrics=f"Sample lyrics for song {i}"
                    )
            
            # 생성
            try:
                result = self.pipeline.generate(input_data, GenerationConfig())
                
                # 파일 저장
                filename = f"latent_{task.lower()}_{i:06d}.npy"
                filepath = output_dir / filename
                
                self.pipeline.save_latents(
                    latents=result['latents'],
                    output_path=filepath,
                    metadata=result.get('metadata')
                )
                
                generated_files.append(str(filepath))
                
                # 진행상황 로깅
                if (i + 1) % save_interval == 0:
                    logger.info(f"Generated {i + 1}/{num_samples} latent samples")
                
            except Exception as e:
                logger.error(f"Failed to generate sample {i}: {e}")
                continue
        
        logger.info(f"Dataset generation completed: {len(generated_files)} files")
        return generated_files