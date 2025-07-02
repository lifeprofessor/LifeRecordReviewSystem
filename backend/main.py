from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr
import os
from typing import Optional, List, Dict
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import ssl
import gc
import re
import uuid
from datetime import datetime, timedelta
import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.poolmanager import PoolManager
import asyncio
from asyncio import Semaphore
import threading
import time
import numpy as np
import atexit
import signal

# ONNX Runtime 관련 import (안전한 방식)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("⚠️ ONNX Runtime이 설치되지 않았습니다.")
    ONNX_AVAILABLE = False

# Transformers import (안전한 방식)
try:
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers 패키지가 설치되지 않았습니다.")
    TRANSFORMERS_AVAILABLE = False

# Optimum import (안전한 방식)
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    OPTIMUM_AVAILABLE = True
except ImportError:
    print("⚠️ Optimum 패키지가 설치되지 않았습니다.")
    OPTIMUM_AVAILABLE = False

# pynvml import (선택사항)
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    print("📝 pynvml이 설치되지 않았습니다. GPU 정보 표시가 제한됩니다.")
    PYNVML_AVAILABLE = False

# PyTorch import (로컬 모델용)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch가 설치되지 않았습니다. 로컬 모델 사용이 제한됩니다.")
    TORCH_AVAILABLE = False

# httpx 모듈 전역 패치
import httpx

_original_client_init = httpx.Client.__init__
_original_async_client_init = httpx.AsyncClient.__init__

def _patched_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    kwargs.setdefault('timeout', 60.0)
    return _original_client_init(self, *args, **kwargs)

def _patched_async_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    kwargs.setdefault('timeout', 60.0)
    return _original_async_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_client_init
httpx.AsyncClient.__init__ = _patched_async_client_init

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CustomHTTPAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=ssl.PROTOCOL_TLS,
            ssl_context=ctx
        )

session = requests.Session()
adapter = CustomHTTPAdapter()
session.mount("https://", adapter)
session.mount("http://", adapter)
requests.Session = lambda: session

cert_path = "C:\\cert\\sdj_ssl.crt"
if os.path.exists(cert_path):
    os.environ['CURL_CA_BUNDLE'] = cert_path
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
    os.environ['SSL_CERT_FILE'] = cert_path
    os.environ['SSL_CERT_DIR'] = os.path.dirname(cert_path)
else:
    print(f"Warning: Certificate file not found at {cert_path}")
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_CERT_FILE'] = ''
    os.environ['SSL_CERT_DIR'] = ''

os.environ['PYTHONHTTPSVERIFY'] = '0'

load_dotenv()

# Configuration constants
# 로컬 모델 경로 (절대 경로로 변경)
LOCAL_MODEL_PATH = os.path.abspath("./model_files")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEARCH_K = 3
CHROMA_DB_DIR = "./chroma_db"
MODEL_CACHE_DIR = "./model_cache"
ONNX_MODEL_DIR = "./onnx_models"
SESSION_TIMEOUT = timedelta(hours=1)

# ONNX 강제 사용 플래그 (RTX 5070 sm_120 문제 해결용)
FORCE_ONNX_MODE = os.getenv("FORCE_ONNX_MODE", "true").lower() == "true"

# 로컬 모델만 사용 (온라인 모델 제거)
EMBEDDING_MODEL = LOCAL_MODEL_PATH

MAX_CONCURRENT_GPU_REQUESTS = 5
BATCH_SIZE = 16
gpu_semaphore = Semaphore(MAX_CONCURRENT_GPU_REQUESTS)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class LoadDocumentRequest(BaseModel):
    area: str
    academic_level: str

class ReviewRequest(BaseModel):
    statement: str
    session_id: str

class ReviewResponse(BaseModel):
    evaluation: str
    feedback: str
    suggestion: str
    suggestion_length: int

class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    area: str
    academic_level: str

# Global variables
tokenizer = None
ort_session = None
sessions: Dict[str, dict] = {}

def check_cuda_availability():
    """CUDA 사용 가능 여부 확인 (안전한 방식)"""
    if not ONNX_AVAILABLE:
        return False, None, None
        
    try:
        providers = ort.get_available_providers()
        cuda_available = 'CUDAExecutionProvider' in providers
        
        if cuda_available:
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory = int(memory_info.total) / (1024**3)
                    return True, gpu_name, total_memory
                except Exception as e:
                    print(f"pynvml 오류: {str(e)}")
                    return True, "NVIDIA GPU", "알 수 없음"
            else:
                return True, "NVIDIA GPU", "알 수 없음"
        else:
            return False, None, None
    except Exception as e:
        print(f"CUDA 확인 중 오류: {str(e)}")
        return False, None, None

def setup_onnx_providers():
    """ONNX Runtime 프로바이더 설정"""
    if not ONNX_AVAILABLE:
        raise Exception("ONNX Runtime이 설치되지 않았습니다.")
        
    cuda_available, gpu_name, gpu_memory = check_cuda_availability()
    
    if cuda_available:
        print(f"🚀 CUDA GPU 사용: {gpu_name}")
        if gpu_memory != "알 수 없음":
            print(f"📊 GPU 메모리: {gpu_memory:.1f}GB")
        print(f"👥 최대 동시 사용자: {MAX_CONCURRENT_GPU_REQUESTS}")
        print(f"📦 배치 처리 크기: {BATCH_SIZE}")
        
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': int(2 * 1024 * 1024 * 1024),
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
    else:
        print("🖥️ CPU 모드로 실행합니다.")
        providers = ['CPUExecutionProvider']
    
    return providers

def download_and_cache_model():
    """로컬 모델을 직접 ONNX로 변환 (SentenceTransformer 우회)"""
    global tokenizer, ort_session
    
    # 로컬 모델 존재 확인
    if not os.path.exists(EMBEDDING_MODEL):
        raise HTTPException(status_code=500, detail=f"로컬 모델을 찾을 수 없습니다: {EMBEDDING_MODEL}")
    
    try:
        print("🔄 로컬 모델 직접 ONNX 변환 중...")
        print(f"📍 모델 경로: {EMBEDDING_MODEL}")
        
        # ONNX 필수 패키지 확인
        if not ONNX_AVAILABLE or not OPTIMUM_AVAILABLE:
            raise Exception("ONNX Runtime 또는 Optimum 패키지가 설치되지 않았습니다.")
        
        # 디렉토리 생성
        for dir_path in [MODEL_CACHE_DIR, ONNX_MODEL_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # 토크나이저 직접 로드
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, local_files_only=True)
        
        # ONNX 모델 로드 또는 생성
        onnx_model_path = os.path.join(ONNX_MODEL_DIR, "model.onnx")
        
        if not os.path.exists(onnx_model_path):
            print("📦 Transformers 모델을 직접 ONNX로 변환 중...")
            
            # *** 핵심: SentenceTransformer 대신 직접 변환 ***
            from transformers.models.auto.modeling_auto import AutoModel
            
            # PyTorch 모델 로드
            pytorch_model = AutoModel.from_pretrained(EMBEDDING_MODEL, local_files_only=True)
            
            # 임시 디렉토리에 모델 저장 (ONNX 변환용)
            temp_model_dir = os.path.join(MODEL_CACHE_DIR, "temp_for_onnx")
            os.makedirs(temp_model_dir, exist_ok=True)
            
            pytorch_model.save_pretrained(temp_model_dir)
            tokenizer.save_pretrained(temp_model_dir)
            
            # ONNX 변환
            model = ORTModelForFeatureExtraction.from_pretrained(
                temp_model_dir,
                export=True,
                local_files_only=True
            )
            model.save_pretrained(ONNX_MODEL_DIR)
            
            # 임시 디렉토리 정리
            import shutil
            shutil.rmtree(temp_model_dir)
            
        else:
            print("📁 기존 ONNX 모델 로드 중...")
            model = ORTModelForFeatureExtraction.from_pretrained(ONNX_MODEL_DIR)
        
        # ONNX Runtime 세션 생성
        providers = setup_onnx_providers()
        
        onnx_files = [f for f in os.listdir(ONNX_MODEL_DIR) if f.endswith('.onnx')]
        if onnx_files:
            actual_onnx_path = os.path.join(ONNX_MODEL_DIR, onnx_files[0])
        else:
            actual_onnx_path = model.model_path
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        ort_session = ort.InferenceSession(
            actual_onnx_path,
            sess_options,
            providers=providers
        )
        
        print("✅ 로컬 모델 ONNX+GPU 변환 완료!")
        print(f"🔧 사용 중인 프로바이더: {ort_session.get_providers()}")
        
        return tokenizer, ort_session
        
    except Exception as e:
        print(f"모델 로드 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"직접 ONNX 변환 중 오류: {str(e)}")

async def get_embeddings_batch_unified(texts, tokenizer, model_or_session):
    """PyTorch와 ONNX 모두 지원하는 통합 임베딩 생성"""
    async with gpu_semaphore:
        try:
            all_embeddings = []
            
            # ONNX Runtime 세션인지 PyTorch 모델인지 확인
            is_onnx = hasattr(model_or_session, 'run')  # ONNX 세션은 run 메서드가 있음
            
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                
                if is_onnx:
                    # ONNX Runtime 방식
                    inputs = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="np"
                    )
                    
                    ort_inputs = {
                        'input_ids': inputs['input_ids'].astype(np.int64),
                        'attention_mask': inputs['attention_mask'].astype(np.int64)
                    }
                    
                    outputs = model_or_session.run(None, ort_inputs)
                    last_hidden_state = outputs[0]
                    
                    attention_mask = inputs['attention_mask']
                    mask_expanded = np.expand_dims(attention_mask, axis=-1)
                    mask_expanded = np.broadcast_to(mask_expanded, last_hidden_state.shape)
                    
                    sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
                    sum_mask = np.sum(mask_expanded, axis=1)
                    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
                    
                    embeddings = sum_embeddings / sum_mask
                    
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms = np.clip(norms, a_min=1e-9, a_max=None)
                    embeddings = embeddings / norms
                    
                    batch_embeddings = embeddings.tolist()
                    
                else:
                    # PyTorch 방식
                    import torch
                    
                    device = next(model_or_session.parameters()).device
                    
                    inputs = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    # GPU로 이동
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model_or_session(**inputs)
                        last_hidden_state = outputs.last_hidden_state
                        
                        # Mean pooling
                        attention_mask = inputs['attention_mask']
                        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                        
                        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                        sum_mask = torch.sum(mask_expanded, 1)
                        sum_mask = torch.clamp(sum_mask, min=1e-9)
                        
                        embeddings = sum_embeddings / sum_mask
                        
                        # L2 정규화
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        
                        # CPU로 이동 후 리스트 변환
                        batch_embeddings = embeddings.cpu().tolist()
                
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"임베딩 생성 중 오류: {str(e)}")

async def get_embeddings_unified(text, tokenizer, model_or_session):
    """단일 텍스트용 통합 임베딩"""
    results = await get_embeddings_batch_unified([text], tokenizer, model_or_session)
    return results[0]

class UnifiedEmbeddingFunction(Embeddings):
    def __init__(self, tokenizer, model_or_session):
        self.tokenizer = tokenizer
        self.model_or_session = model_or_session

    def embed_documents(self, texts):
        """동기 방식 문서 임베딩 (ChromaDB 호환)"""
        try:
            return self._get_embeddings_batch_sync(texts)
        except Exception as e:
            print(f"문서 임베딩 오류: {str(e)}")
            raise

    def embed_query(self, text):
        """동기 방식 쿼리 임베딩 (ChromaDB 호환)"""
        try:
            results = self._get_embeddings_batch_sync([text])
            return results[0]
        except Exception as e:
            print(f"쿼리 임베딩 오류: {str(e)}")
            raise

    def _get_embeddings_batch_sync(self, texts):
        """동기 방식 배치 임베딩 생성"""
        try:
            all_embeddings = []
            
            # ONNX Runtime 세션인지 PyTorch 모델인지 확인
            is_onnx = hasattr(self.model_or_session, 'run')
            
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                
                if is_onnx:
                    # ONNX Runtime 방식
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="np"
                    )
                    
                    ort_inputs = {
                        'input_ids': inputs['input_ids'].astype(np.int64),
                        'attention_mask': inputs['attention_mask'].astype(np.int64)
                    }
                    
                    outputs = self.model_or_session.run(None, ort_inputs)
                    last_hidden_state = outputs[0]
                    
                    attention_mask = inputs['attention_mask']
                    mask_expanded = np.expand_dims(attention_mask, axis=-1)
                    mask_expanded = np.broadcast_to(mask_expanded, last_hidden_state.shape)
                    
                    sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
                    sum_mask = np.sum(mask_expanded, axis=1)
                    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
                    
                    embeddings = sum_embeddings / sum_mask
                    
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms = np.clip(norms, a_min=1e-9, a_max=None)
                    embeddings = embeddings / norms
                    
                    batch_embeddings = embeddings.tolist()
                    
                else:
                    # PyTorch 방식
                    import torch
                    
                    device = next(self.model_or_session.parameters()).device
                    
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    # GPU로 이동
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model_or_session(**inputs)
                        last_hidden_state = outputs.last_hidden_state
                        
                        # Mean pooling
                        attention_mask = inputs['attention_mask']
                        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                        
                        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                        sum_mask = torch.sum(mask_expanded, 1)
                        sum_mask = torch.clamp(sum_mask, min=1e-9)
                        
                        embeddings = sum_embeddings / sum_mask
                        
                        # L2 정규화
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        
                        # CPU로 이동 후 리스트 변환
                        batch_embeddings = embeddings.cpu().tolist()
                
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            raise Exception(f"임베딩 생성 중 오류: {str(e)}")

def cleanup_expired_sessions():
    current_time = datetime.now()
    expired_sessions = [
        session_id for session_id, session_data in sessions.items()
        if current_time - session_data['created_at'] > SESSION_TIMEOUT
    ]
    
    for session_id in expired_sessions:
        try:
            # 1. vectorstore 객체 먼저 정리
            if session_id in sessions and 'vectorstore' in sessions[session_id]:
                vectorstore = sessions[session_id]['vectorstore']
                if hasattr(vectorstore, '_client') and vectorstore._client:
                    try:
                        vectorstore._client.reset()
                    except:
                        pass
                del vectorstore
            
            # 2. 세션에서 제거
            if session_id in sessions:
                del sessions[session_id]
            
            # 3. 파일 시스템에서 삭제 (재시도 메커니즘 포함)
            session_dir = os.path.join(CHROMA_DB_DIR, session_id)
            if os.path.exists(session_dir):
                import shutil
                import time
                
                # Windows에서의 파일 삭제 재시도
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # 가비지 컬렉션 강제 실행
                        import gc
                        gc.collect()
                        
                        # 잠시 대기 후 삭제 시도
                        if attempt > 0:
                            time.sleep(0.5)
                        
                        shutil.rmtree(session_dir)
                        print(f"✅ 만료된 세션 디렉토리 삭제 완료: {session_id}")
                        break
                        
                    except PermissionError as e:
                        if attempt == max_retries - 1:
                            print(f"⚠️ 세션 디렉토리 삭제 실패 (권한 문제): {session_id} - {str(e)}")
                            # 삭제 실패 시에도 세션은 메모리에서 제거했으므로 계속 진행
                        else:
                            print(f"🔄 세션 디렉토리 삭제 재시도 중... ({attempt + 1}/{max_retries}): {session_id}")
                    except Exception as e:
                        print(f"❌ 세션 디렉토리 삭제 중 예외 발생: {session_id} - {str(e)}")
                        break
                        
        except Exception as e:
            print(f"❌ 세션 정리 중 예외 발생: {session_id} - {str(e)}")
            # 개별 세션 정리 실패 시에도 다른 세션들은 계속 처리
            continue

@app.post("/api/load-documents")
async def load_documents(request: LoadDocumentRequest):
    try:
        cleanup_expired_sessions()
        
        area_map = {
            "자율/자치활동 특기사항": "self_governance_guidelines",
            "진로활동 특기사항": "career_activity_guidelines"
        }
        
        if request.area not in area_map:
            raise HTTPException(status_code=400, detail=f"Invalid area selected: {request.area}")
        
        directory = f"data/{area_map[request.area]}"
        if not os.path.exists(directory):
            raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
        
        if not os.listdir(directory):
            raise HTTPException(status_code=404, detail=f"Directory is empty: {directory}")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        session_db_dir = os.path.join(CHROMA_DB_DIR, session_id)
        
        # Load documents
        documents = []
        for file_path in os.listdir(directory):
            try:
                if file_path.endswith('.md'):
                    with open(os.path.join(directory, file_path), 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append(Document(page_content=content, metadata={"source": file_path}))
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
        
        if len(documents) == 0:
            raise HTTPException(status_code=404, detail=f"No markdown files found in {directory}")
        
        # Split text
        text_splitter = MarkdownTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        
        # Initialize models if needed
        if tokenizer is None or ort_session is None:
            try:
                download_and_cache_model()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")
        
        # 통합 임베딩 함수 사용 (PyTorch/ONNX 자동 감지)
        embedding_function = UnifiedEmbeddingFunction(tokenizer, ort_session)
        
        try:
            os.makedirs(session_db_dir, exist_ok=True)
            
            # 사용 중인 모델 타입 확인
            is_local_model = (os.path.exists(EMBEDDING_MODEL) and 
                            os.path.isdir(EMBEDDING_MODEL) and
                            os.path.exists(os.path.join(EMBEDDING_MODEL, "config.json")))
            
            # 가속 타입 결정
            if FORCE_ONNX_MODE:
                acceleration_type = "ONNX Runtime (GPU)"
            elif hasattr(ort_session, 'run'):  # ONNX 세션
                acceleration_type = "ONNX Runtime"
            else:  # PyTorch 모델
                acceleration_type = "PyTorch (로컬)"
            
            print(f"📊 {acceleration_type}로 문서 {len(splits)}개 처리 중... (사용자: {len(sessions)+1}명)")
            start_time = time.time()
            
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding_function,
                persist_directory=session_db_dir
            )
            
            end_time = time.time()
            print(f"⚡ {acceleration_type} 문서 처리 완료: {end_time - start_time:.1f}초")
            
            # Store session information
            sessions[session_id] = {
                'vectorstore': vectorstore,
                'created_at': datetime.now(),
                'area': request.area,
                'academic_level': request.academic_level
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating vectorstore: {str(e)}")
        
        # 메모리 정리
        gc.collect()
        
        active_sessions = len(sessions)
        
        return {
            "status": "success",
            "message": f"Documents loaded successfully with {acceleration_type}",
            "session_id": session_id,
            "server_info": {
                "active_sessions": active_sessions,
                "processing_time": f"{end_time - start_time:.1f}s",
                "acceleration": acceleration_type,
                "model_type": "로컬 모델" if is_local_model else "온라인 모델"
            }
        }

    except Exception as e:
        import traceback
        print("--- LOAD DOCUMENTS ENDPOINT ERROR ---")
        traceback.print_exc()
        print("-------------------------------------")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/api/review")
async def review_statement(request: ReviewRequest):
    cleanup_expired_sessions()  # 만료된 세션 정리
    
    if request.session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid or expired session ID")
    
    session_data = sessions[request.session_id]
    vectorstore = session_data['vectorstore']
    
    if not request.statement:
        raise HTTPException(status_code=400, detail="Statement is required")
    
    try:
        template = """
당신은 고등학교 생기부 특기사항 작성 전문가입니다.

아래 입력된 문장을 기반으로 다음 세 가지 항목을 평가해 주세요:

① 적합성 평가: 해당 문장이 작성요령에 적합한지 간단히 판단해 주세요.
② 검토 의견: 문장의 장점, 부족한 점, 개선 포인트를 구체적으로 설명해 주세요.
③ 개선 제안: 작성요령을 고려하여 문장을 더 나은 형태의 500자로 수정해 주세요

입력 문장:
{question}

※ 응답은 반드시 위 순서와 번호(①②③)를 포함하여 간결하고 명확하게 출력해 주세요.
        """

        print("=== REVIEW API DEBUG ===")
        print(f"Session ID: {request.session_id}")
        print(f"Statement: {request.statement}")
        print(f"Vectorstore type: {type(vectorstore)}")
        
        try:
            chain = create_chain(vectorstore)
            print("Chain created successfully")
        except Exception as chain_error:
            print(f"Chain creation error: {str(chain_error)}")
            raise HTTPException(status_code=500, detail=f"Chain creation failed: {str(chain_error)}")
        
        print("Claude API 호출 프롬프트 전체:")
        print(template.replace("{question}", request.statement))
        
        try:
            response = chain.invoke(request.statement)
            print("Chain invoke completed")
        except Exception as invoke_error:
            print(f"Chain invoke error: {str(invoke_error)}")
            raise HTTPException(status_code=500, detail=f"Chain invoke failed: {str(invoke_error)}")
        
        # Claude 응답에서 content 추출
        result = getattr(response, "content", None)
        if not result and isinstance(response, dict):
            result = response.get("content", "")
        elif not result:
            result = str(response)

        print("Claude API 응답 원문:\n", result)

        # 각 항목 파싱
        eval_part, feedback_part, suggestion_part = "", "", ""
        try:
            parts = result.split("①")[1].split("②")
            eval_part = parts[0].strip()
            feedback_suggestion = parts[1].split("③")
            feedback_part = feedback_suggestion[0].strip()
            suggestion_part = feedback_suggestion[1].strip()
        except Exception as parse_error:
            print(f"Parsing error: {str(parse_error)}")
            eval_part = result.strip()

        # 파싱 후
        eval_part = remove_heading(eval_part, "적합성 평가")
        feedback_part = remove_heading(feedback_part, "검토 의견")
        suggestion_part = remove_heading(suggestion_part, "개선 제안")

        # 가독성 개선
        eval_part = prettify_evaluation(eval_part)
        feedback_part = prettify_feedback(feedback_part)
        suggestion_part = prettify_suggestion(suggestion_part)

        # 500자 제한 (공백 포함)
        suggestion_part = suggestion_part[:500]
        suggestion_length = len(suggestion_part)

        print("=== REVIEW API SUCCESS ===")
        return {
            "evaluation": eval_part,
            "feedback": feedback_part,
            "suggestion": suggestion_part,
            "suggestion_length": suggestion_length
        }

    except Exception as e:
        import traceback
        print("=== REVIEW API ERROR ===")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        print("========================")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def list_sessions():
    """현재 활성화된 세션 목록을 반환합니다."""
    cleanup_expired_sessions()
    return [
        SessionInfo(
            session_id=session_id,
            created_at=session_data['created_at'],
            area=session_data['area'],
            academic_level=session_data['academic_level']
        )
        for session_id, session_data in sessions.items()
    ]

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # vectorstore 객체 먼저 정리
        if 'vectorstore' in sessions[session_id]:
            vectorstore = sessions[session_id]['vectorstore']
            if hasattr(vectorstore, '_client') and vectorstore._client:
                try:
                    vectorstore._client.reset()
                except:
                    pass
            del vectorstore
        
        # 메모리에서 세션 제거
        del sessions[session_id]
        
        # 파일 시스템에서 삭제 (비동기적으로 처리)
        session_dir = os.path.join(CHROMA_DB_DIR, session_id)
        if os.path.exists(session_dir):
            import asyncio
            import shutil
            
            async def cleanup_directory():
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        await asyncio.sleep(0.1)  # 짧은 대기
                        shutil.rmtree(session_dir)
                        break
                    except PermissionError:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.5)
                        else:
                            print(f"⚠️ 세션 디렉토리 삭제 지연됨: {session_id}")
            
            # 백그라운드에서 삭제 작업 수행
            asyncio.create_task(cleanup_directory())
        
        return {"status": "success", "message": "Session deleted"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

def remove_heading(text, heading):
    # "개선 제안", "개선 제안:", "개선 제안 " 등으로 시작하면 제거
    for h in [heading, f"{heading}:", f"{heading} "]:
        if text.strip().startswith(h):
            return text.strip()[len(h):].lstrip(": \n\"")
    return text.strip()

def prettify_bullet(text, emoji="•"):
    # '- '로 시작하는 항목을 이모지로 변환
    lines = text.split('\n')
    pretty_lines = []
    for line in lines:
        if line.strip().startswith('- '):
            pretty_lines.append(f"{emoji} {line.strip()[2:]}")
        else:
            pretty_lines.append(line.strip())
    return '\n'.join(pretty_lines)

def prettify_feedback(text):
    # 장점/부족한 점을 이모지와 함께, 리스트는 들여쓰기
    text = text.replace("장점:", "\n💡 ")
    text = text.replace("부족한 점:", "\n⚠️ ")
    text = text.replace("개선필요:", "\n📝 ")
    text = text.replace("개선점:", "\n📝 ")
    text = prettify_bullet(text, emoji="👉")
    return text.strip()

def prettify_evaluation(text):
    return prettify_bullet(text, emoji="✅")

def prettify_suggestion(text):
    # 쌍따옴표 제거
    return text.replace('"', '').strip()

def create_chain(vectorstore):
    try:
        print("Creating retriever...")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": SEARCH_K}
        )
        print("Retriever created successfully")
        
        template = """
        당신은 고등학교 생기부 특기사항 작성 전문가입니다.

        아래 입력된 문장을 기반으로 다음 세 가지 항목을 평가해 주세요:

        ① 적합성 평가: 해당 문장이 작성요령에 적합한지 간단히 판단해 주세요.
        ② 검토 의견: 문장의 장점, 부족한 점, 개선 포인트를 구체적으로 설명해 주세요.
        ③ 개선 제안: 작성요령을 고려하여 문장을 더 나은 형태의 500자로 수정해 주세요.

        입력 문장:
        {question}

        ※ 응답은 반드시 위 순서와 번호(①②③)를 포함하여 간결하고 명확하게 출력해 주세요.
        """

        print("Creating prompt template...")
        prompt = ChatPromptTemplate.from_template(template)
        print("Prompt template created successfully")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not found")
        
        print("Creating ChatAnthropic model...")
        
        # SSL 컨텍스트 생성 및 검증 비활성화
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # 전역 SSL 컨텍스트 설정
        ssl._create_default_https_context = lambda: ssl_context
        
        # 환경 변수를 통해 SSL 검증 비활성화
        os.environ['ANTHROPIC_VERIFY_SSL'] = 'false'
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_CERT_FILE'] = ''
        
        model = ChatAnthropic(
            model_name="claude-3-5-sonnet-20241022",
            temperature=0,
            api_key=SecretStr(api_key),
            max_tokens_to_sample=2048,
            timeout=60,
            stop=None
        )
        print("ChatAnthropic model created successfully")
        
        # Anthropic 클라이언트의 내부 httpx 클라이언트 수정
        if hasattr(model, '_client') and hasattr(model._client, '_client'):
            # 기존 클라이언트 설정 복사
            old_client = model._client._client
            # SSL 검증 비활성화된 새 클라이언트 생성
            new_client = httpx.Client(
                verify=False,
                timeout=60.0,
                headers=old_client.headers,
                cookies=old_client.cookies,
                auth=old_client.auth,
                follow_redirects=old_client.follow_redirects
            )
            # 내부 클라이언트 교체
            model._client._client = new_client
            print("Anthropic client SSL verification disabled successfully")
        
        print("Creating chain...")
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
        )
        print("Chain created successfully")
        
        return chain
    except Exception as e:
        import traceback
        print("=== CREATE CHAIN ERROR ===")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        print("==========================")
        raise HTTPException(status_code=500, detail=f"Chain 생성 중 오류 발생: {str(e)}")

def cleanup_all_sessions():
    """서버 종료 시 모든 세션 정리"""
    print("🧹 서버 종료 중 - 모든 세션 정리...")
    for session_id in list(sessions.keys()):
        try:
            if 'vectorstore' in sessions[session_id]:
                vectorstore = sessions[session_id]['vectorstore']
                if hasattr(vectorstore, '_client') and vectorstore._client:
                    vectorstore._client.reset()
        except:
            pass
    sessions.clear()

# 서버 종료 시 정리 함수 등록
atexit.register(cleanup_all_sessions)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 