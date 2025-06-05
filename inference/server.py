# lyro/inference/server.py
"""
LYRO Inference Server
FastAPI 기반 RESTful API 서버
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import tempfile
import uuid
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# LYRO 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssm.model import LyroSSMUNet, TaskController
from ssm.flow_matching import LyroFlowMatching, FlowConfig
from dcae.model import create_cqt_ssm_dcae  # ✅ 수정: CQT 모델 사용
from dataset.tokenizer import LyroTokenizer


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="LYRO Music Generation API",
    description="AI-powered music generation with lyrics, style control, and editing capabilities",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 모델 인스턴스
class ModelManager:
    """모델 관리 클래스"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_loaded = False
        self.ssm_model = None
        self.dcae_model = None
        self.flow_matching = None
        self.tokenizer = None
        self.temp_dir = Path("temp_outputs")
        self.temp_dir.mkdir(exist_ok=True)
        
        # 비동기 실행을 위한 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def load_models(self, ssm_checkpoint: str, dcae_checkpoint: str):
        """모델 로드"""
        if self.models_loaded:
            return
            
        logger.info("Loading models...")
        
        # SSM 모델
        self.ssm_model = LyroSSMUNet(
            input_channels=8,
            hidden_dims=[128, 128, 256, 256, 512],
            mamba_layers=[2, 2, 3, 3, 4],
            max_seq_len=8192  # ✅ 수정: 6144 -> 8192
        ).to(self.device)
        
        # SSM 체크포인트 로드
        ssm_ckpt = torch.load(ssm_checkpoint, map_location=self.device)
        self.ssm_model.load_state_dict(ssm_ckpt['model_state_dict'])
        self.ssm_model.eval()
          # DCAE 모델 (CQT 모델 사용)
        self.dcae_model = create_cqt_ssm_dcae(
            sample_rate=44100,
            latent_channels=8,
            model_size="base",  # CQT 모델 크기
            use_torch_compile=False,  # 서버에서는 컴파일 비활성화
            use_mixed_precision=True
        ).to(self.device)
        
        # DCAE 체크포인트 로드
        dcae_ckpt = torch.load(dcae_checkpoint, map_location=self.device)
        self.dcae_model.load_state_dict(dcae_ckpt['model_state_dict'])
        self.dcae_model.eval()
        
        # Flow Matching
        flow_config = FlowConfig()
        self.flow_matching = LyroFlowMatching(  # ✅ 수정: FlowMatching -> LyroFlowMatching
            model=self.ssm_model,
            scheduler_type="cosine",
            solver_type="heun", 
            sigma=1e-4,
            flow_type="rectified"
        )
        
        # 토크나이저
        self.tokenizer = LyroTokenizer()
        
        self.models_loaded = True
        logger.info("Models loaded successfully!")
        
    async def generate_async(self, *args, **kwargs):
        """비동기 생성 래퍼"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate, *args, **kwargs)
    
    def generate(
        self,
        task: str,
        lyrics: Optional[str] = None,
        style_prompt: Optional[str] = None,
        reference_audio: Optional[np.ndarray] = None,
        duration: float = 30.0,
        steps: Optional[int] = None,
        guidance_scale: float = 1.5
    ) -> Dict:
        """음악 생성"""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded")
            
        try:
            with torch.no_grad():
                # 조건 준비
                conditions = TaskController.create_conditions(
                    task_type=task,
                    lyrics=self._encode_lyrics(lyrics) if lyrics else None,
                    style_prompt=self._encode_style(style_prompt) if style_prompt else None,
                    icl_reference=self._encode_reference(reference_audio) if reference_audio is not None else None,
                    device=self.device  # ✅ 추가: device 전달
                )
                  # Flow Matching 생성 (CQT 압축율 반영: ~32x)
                # ✅ 수정: 32x 압축율 반영
                duration_samples = min(
                    int(duration * 44100 / (512 * 32)),
                    self.ssm_model.max_seq_len
                )
                
                if duration_samples <= 0:
                    duration_samples = 64  # 최소값
                    
                shape = (1, 8, duration_samples)
                
                # 생성 설정
                if steps is None:
                    steps = 4 if task == 'INPAINT' else 8 if task == 'INST' else 12
                    
                generated_latent, _ = self.flow_matching.generate(
                    shape=shape,
                    conditions=conditions,
                    num_steps=steps,  # ✅ 수정: steps -> num_steps
                    cfg_scale=guidance_scale
                )
                
                # 안전성 검증
                if torch.isnan(generated_latent).any():
                    raise ValueError("Generated latent contains NaN")
                    
                if torch.isinf(generated_latent).any():
                    raise ValueError("Generated latent contains Inf")
                
                # DCAE 디코딩 (더미 skip_features 생성)
                dummy_skip_features = [torch.zeros_like(generated_latent) for _ in range(5)]
                audio = self.dcae_model.decode(generated_latent, dummy_skip_features)
                
                # 오디오 후처리
                audio = torch.clamp(audio, -1.0, 1.0)  # 클리핑 방지
                
                # 임시 파일 저장
                output_id = str(uuid.uuid4())
                output_path = self.temp_dir / f"{output_id}.wav"
                
                torchaudio.save(
                    output_path,
                    audio[0].cpu(),
                    sample_rate=44100
                )
                
                return {
                    'output_id': output_id,
                    'output_path': str(output_path),
                    'duration': duration,
                    'task': task,
                    'generation_steps': steps
                }
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            # 더미 출력 생성
            output_id = str(uuid.uuid4())
            output_path = self.temp_dir / f"{output_id}.wav"
            
            # 무음 생성
            dummy_audio = torch.zeros(2, int(duration * 44100))
            torchaudio.save(output_path, dummy_audio, sample_rate=44100)
            
            return {
                'output_id': output_id,
                'output_path': str(output_path),
                'duration': duration,
                'task': task,
                'generation_steps': steps or 8,
                'error': str(e)
            }
            
    def _encode_lyrics(self, lyrics: str) -> torch.Tensor:
        """가사 인코딩"""
        tokens = self.tokenizer.encode_text(lyrics)
        return torch.tensor([tokens]).to(self.device)
    
    def _encode_style(self, style_prompt: str) -> torch.Tensor:
        """스타일 프롬프트 인코딩"""
        # 간단한 텍스트 임베딩 (실제로는 더 복잡한 처리 필요)
        tokens = self.tokenizer.encode_text(style_prompt)
        # 고정 크기 벡터로 변환
        style_vec = torch.zeros(512)
        for i, token in enumerate(tokens[:50]):  # 최대 50 토큰
            if i * 10 + 10 <= 512:  # 범위 검증
                style_vec[i * 10:(i + 1) * 10] = min(token / 1000.0, 1.0)  # 정규화
        return style_vec.unsqueeze(0).to(self.device)
    
    def _encode_reference(self, audio: np.ndarray) -> torch.Tensor:
        """참조 오디오 인코딩"""
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)
        elif audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.repeat(2, 1)
        elif audio_tensor.shape[0] > 2:
            audio_tensor = audio_tensor[:2]
            
        audio_tensor = audio_tensor.unsqueeze(0).to(self.device)
        
        # DCAE 인코딩
        latent, _ = self.dcae_model.encode(audio_tensor)  # ✅ 수정: skip_features 처리
        return latent


# 모델 매니저 인스턴스
model_manager = ModelManager()


# API 엔드포인트들

class GenerateRequest(BaseModel):
    """생성 요청 모델"""
    task: str = "SONG"
    lyrics: Optional[str] = None
    style_prompt: Optional[str] = None
    duration: float = 30.0
    steps: Optional[int] = None
    guidance_scale: float = 1.5


class EditRequest(BaseModel):
    """편집 요청 모델"""
    audio_id: str
    mask_start: float
    mask_end: float
    edit_prompt: str
    steps: int = 4


class ExtendRequest(BaseModel):
    """연장 요청 모델"""
    audio_id: str
    extend_duration: float
    style_prompt: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    # 환경 변수에서 체크포인트 경로 읽기
    ssm_checkpoint = os.getenv("SSM_CHECKPOINT", "ssm/checkpoints/best_model_stage_E.pt")
    dcae_checkpoint = os.getenv("DCAE_CHECKPOINT", "dcae/checkpoints/best_model.pt")
    
    try:
        model_manager.load_models(ssm_checkpoint, dcae_checkpoint)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # 서버는 계속 실행하되 에러 상태로 표시
        pass


@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "status": "online",
        "models_loaded": model_manager.models_loaded,
        "device": str(model_manager.device)
    }


@app.post("/generate")
async def generate_music(request: GenerateRequest):
    """음악 생성 엔드포인트"""
    try:
        # 입력 검증
        if request.task not in ["SONG", "INST", "COVER"]:
            raise HTTPException(status_code=400, detail="Invalid task type")
            
        if request.task == "SONG" and not request.lyrics:
            raise HTTPException(status_code=400, detail="Lyrics required for SONG task")
            
        # 비동기 생성
        result = await model_manager.generate_async(
            task=request.task,
            lyrics=request.lyrics,
            style_prompt=request.style_prompt,
            duration=request.duration,
            steps=request.steps,
            guidance_scale=request.guidance_scale
        )
        
        return JSONResponse(content={
            "status": "success",
            "output_id": result['output_id'],
            "duration": result['duration'],
            "task": result['task'],
            "generation_steps": result['generation_steps'],
            "error": result.get('error')  # 에러가 있으면 포함
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_cover")
async def generate_cover(
    reference_audio: UploadFile = File(...),
    lyrics: Optional[str] = Form(None),
    style_prompt: Optional[str] = Form(None),
    reference_duration: float = Form(10.0)
):
    """커버 생성 엔드포인트"""
    try:
        # 참조 오디오 읽기
        audio_bytes = await reference_audio.read()
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        # 오디오 로드
        audio, sr = torchaudio.load(tmp_path)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            audio = resampler(audio)
            
        # 참조 길이 제한
        max_samples = int(reference_duration * 44100)
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
            
        # numpy 변환
        audio_np = audio.numpy()
        
        # 임시 파일 삭제
        os.unlink(tmp_path)
        
        # 커버 생성
        result = await model_manager.generate_async(
            task="COVER",
            lyrics=lyrics,
            style_prompt=style_prompt,
            reference_audio=audio_np,
            duration=30.0,
            guidance_scale=1.5
        )
        
        return JSONResponse(content={
            "status": "success",
            "output_id": result['output_id'],
            "duration": result['duration'],
            "error": result.get('error')
        })
        
    except Exception as e:
        logger.error(f"Cover generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/edit")
async def edit_music(request: EditRequest):
    """음악 편집 엔드포인트"""
    try:
        # 원본 오디오 로드
        audio_path = model_manager.temp_dir / f"{request.audio_id}.wav"
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio not found")
            
        audio, sr = torchaudio.load(audio_path)
        
        # Flow-Edit 적용
        # TODO: Flow-Edit 구현
        
        return JSONResponse(content={
            "status": "success",
            "message": "Edit functionality not yet implemented"
        })
        
    except Exception as e:
        logger.error(f"Edit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extend")
async def extend_music(request: ExtendRequest):
    """음악 연장 엔드포인트"""
    try:
        # 원본 오디오 로드
        audio_path = model_manager.temp_dir / f"{request.audio_id}.wav"
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio not found")
            
        audio, sr = torchaudio.load(audio_path)
        
        # Flow-Extend 적용
        # TODO: Flow-Extend 구현
        
        return JSONResponse(content={
            "status": "success",
            "message": "Extend functionality not yet implemented"
        })
        
    except Exception as e:
        logger.error(f"Extend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{output_id}")
async def download_output(output_id: str):
    """생성된 오디오 다운로드"""
    try:
        file_path = model_manager.temp_dir / f"{output_id}.wav"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename=f"lyro_generated_{output_id}.wav"
        )
        
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cleanup/{output_id}")
async def cleanup_output(output_id: str):
    """임시 파일 정리"""
    try:
        file_path = model_manager.temp_dir / f"{output_id}.wav"
        if file_path.exists():
            file_path.unlink()
            
        return JSONResponse(content={
            "status": "success",
            "message": "File deleted"
        })
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "models_loaded": model_manager.models_loaded,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


if __name__ == "__main__":
    import uvicorn
    
    # 서버 실행
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )