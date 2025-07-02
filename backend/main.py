# ================================================================================================
# 필수 라이브러리 Import 섹션
# ================================================================================================

# FastAPI: 현대적인 웹 API 프레임워크 (Django REST나 Flask보다 빠름)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # 브라우저에서 API 호출 허용

# Pydantic: 데이터 검증 및 타입 체크를 위한 라이브러리
from pydantic import BaseModel, SecretStr  # API 요청/응답 데이터 모델 정의용

# 기본 Python 라이브러리들
import os  # 파일 시스템 및 환경변수 접근
from typing import Optional, List, Dict  # 타입 힌트용 (코드 가독성 향상)

# LangChain: AI 애플리케이션 개발을 위한 프레임워크
from langchain_community.vectorstores import Chroma  # 벡터 데이터베이스 (문서 검색용)
from langchain_anthropic import ChatAnthropic  # Claude AI 모델 연동
from langchain.prompts import ChatPromptTemplate  # AI에게 보낼 프롬프트 템플릿
from langchain.schema.runnable import RunnablePassthrough  # 데이터 전달용 파이프라인
from langchain.schema import Document  # 문서 객체 (텍스트 + 메타데이터)
from langchain.text_splitter import MarkdownTextSplitter  # 마크다운 문서를 청크로 분할
from langchain.embeddings.base import Embeddings  # 임베딩 인터페이스 (텍스트→벡터 변환)

# 환경설정 파일 로드용
from dotenv import load_dotenv  # .env 파일에서 API 키 등 민감정보 읽기

# 보안 및 네트워크 관련
import ssl  # HTTPS 보안 연결 설정
import urllib3  # HTTP 클라이언트 라이브러리
import requests  # HTTP 요청 처리
from requests.adapters import HTTPAdapter  # HTTP 연결 어댑터
from urllib3.util.retry import Retry  # 요청 실패시 재시도 로직
from urllib3.poolmanager import PoolManager  # 연결 풀 관리

# 시스템 및 유틸리티
import gc  # 가비지 컬렉션 (메모리 정리)
import re  # 정규표현식
import uuid  # 고유 ID 생성
from datetime import datetime, timedelta  # 날짜/시간 처리
import asyncio  # 비동기 프로그래밍
from asyncio import Semaphore  # 동시 실행 제한 (GPU 자원 관리용)
import threading  # 멀티스레딩
import time  # 시간 측정
import numpy as np  # 수치 계산 (임베딩 벡터 처리용)
import atexit  # 프로그램 종료시 정리 작업
import signal  # 시스템 신호 처리

# ================================================================================================
# 선택적 AI/ML 라이브러리 Import (try-except로 안전하게 처리)
# ================================================================================================

# ONNX Runtime: 머신러닝 모델을 빠르게 실행하는 라이브러리 (특히 GPU 가속에 유용)
try:
    import onnxruntime as ort  # Microsoft의 AI 모델 최적화 런타임
    ONNX_AVAILABLE = True  # ONNX 사용 가능 플래그
except ImportError:
    print("⚠️ ONNX Runtime이 설치되지 않았습니다.")
    ONNX_AVAILABLE = False

# Transformers: Hugging Face의 자연어처리 모델 라이브러리
try:
    from transformers.models.auto.tokenization_auto import AutoTokenizer  # 텍스트→토큰 변환기
    TRANSFORMERS_AVAILABLE = True  # Transformers 사용 가능 플래그
except ImportError:
    print("⚠️ Transformers 패키지가 설치되지 않았습니다.")
    TRANSFORMERS_AVAILABLE = False

# Optimum: Hugging Face 모델을 ONNX로 변환하는 라이브러리
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction  # 임베딩 모델의 ONNX 버전
    OPTIMUM_AVAILABLE = True  # Optimum 사용 가능 플래그
except ImportError:
    print("⚠️ Optimum 패키지가 설치되지 않았습니다.")
    OPTIMUM_AVAILABLE = False

# pynvml: NVIDIA GPU 정보를 가져오는 라이브러리 (선택사항)
try:
    import pynvml  # GPU 메모리, 이름 등 하드웨어 정보 조회용
    PYNVML_AVAILABLE = True  # GPU 정보 조회 가능 플래그
except ImportError:
    print("📝 pynvml이 설치되지 않았습니다. GPU 정보 표시가 제한됩니다.")
    PYNVML_AVAILABLE = False

# PyTorch: 딥러닝 프레임워크 (로컬 모델 실행용)
try:
    import torch  # 딥러닝 모델 로드 및 실행
    TORCH_AVAILABLE = True  # PyTorch 사용 가능 플래그
except ImportError:
    print("⚠️ PyTorch가 설치되지 않았습니다. 로컬 모델 사용이 제한됩니다.")
    TORCH_AVAILABLE = False

# ================================================================================================
# HTTP 클라이언트 전역 패치 (SSL 검증 비활성화 - 개발/테스트 환경용)
# ================================================================================================

# httpx: 비동기 HTTP 클라이언트 라이브러리 (Claude API 호출용)
import httpx

# 원본 클라이언트 초기화 함수들을 백업
_original_client_init = httpx.Client.__init__  # 동기 클라이언트 원본
_original_async_client_init = httpx.AsyncClient.__init__  # 비동기 클라이언트 원본

def _patched_client_init(self, *args, **kwargs):
    """
    httpx.Client의 패치된 초기화 함수
    - SSL 검증을 비활성화하여 인증서 문제를 회피
    - 타임아웃을 60초로 설정하여 느린 API 응답에 대비
    """
    kwargs['verify'] = False  # SSL 인증서 검증 비활성화
    kwargs.setdefault('timeout', 60.0)  # 기본 타임아웃 60초
    return _original_client_init(self, *args, **kwargs)

def _patched_async_client_init(self, *args, **kwargs):
    """
    httpx.AsyncClient의 패치된 초기화 함수
    - 비동기 클라이언트에도 동일한 SSL/타임아웃 설정 적용
    """
    kwargs['verify'] = False  # SSL 인증서 검증 비활성화
    kwargs.setdefault('timeout', 60.0)  # 기본 타임아웃 60초
    return _original_async_client_init(self, *args, **kwargs)

# httpx 모듈의 기본 동작을 패치된 버전으로 교체
httpx.Client.__init__ = _patched_client_init
httpx.AsyncClient.__init__ = _patched_async_client_init

# SSL 전역 설정: 모든 HTTPS 연결에서 인증서 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

# urllib3 경고 메시지 비활성화 (SSL 검증 비활성화 경고 숨김)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CustomHTTPAdapter(HTTPAdapter):
    """
    사용자 정의 HTTP 어댑터 클래스
    - requests 라이브러리의 SSL 검증을 비활성화
    - 연결 풀 관리를 통해 성능 최적화
    """
    def init_poolmanager(self, connections, maxsize, block=False):
        """
        HTTP 연결 풀 매니저 초기화
        - SSL 컨텍스트를 생성하고 검증을 비활성화
        - 여러 연결을 효율적으로 관리
        """
        ctx = ssl.create_default_context()  # 기본 SSL 컨텍스트 생성
        ctx.check_hostname = False  # 호스트명 검증 비활성화
        ctx.verify_mode = ssl.CERT_NONE  # 인증서 검증 완전 비활성화
        self.poolmanager = PoolManager(
            num_pools=connections,  # 동시 연결 풀 개수
            maxsize=maxsize,  # 각 풀의 최대 연결 수
            block=block,  # 연결 풀이 꽉 찼을 때 대기 여부
            ssl_version=ssl.PROTOCOL_TLS,  # TLS 프로토콜 사용
            ssl_context=ctx  # 위에서 설정한 SSL 컨텍스트 적용
        )

# requests 라이브러리 전역 설정
session = requests.Session()  # 전역 세션 객체 생성
adapter = CustomHTTPAdapter()  # 커스텀 어댑터 인스턴스 생성
session.mount("https://", adapter)  # HTTPS 요청에 커스텀 어댑터 적용
session.mount("http://", adapter)   # HTTP 요청에 커스텀 어댑터 적용
requests.Session = lambda: session  # 새로운 Session 생성시 위의 설정된 세션 반환

# ================================================================================================
# SSL 인증서 및 환경변수 설정
# ================================================================================================

# SSL 인증서 파일 경로 설정 (특정 환경에서 필요한 경우)
cert_path = "C:\\cert\\sdj_ssl.crt"  # 회사/기관 전용 SSL 인증서 경로
if os.path.exists(cert_path):
    # 인증서 파일이 존재하는 경우 관련 환경변수 설정
    os.environ['CURL_CA_BUNDLE'] = cert_path      # cURL 라이브러리용 인증서
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path  # requests 라이브러리용 인증서
    os.environ['SSL_CERT_FILE'] = cert_path       # 일반 SSL 인증서 파일
    os.environ['SSL_CERT_DIR'] = os.path.dirname(cert_path)  # 인증서 디렉토리
else:
    # 인증서 파일이 없는 경우 관련 환경변수를 빈 값으로 설정
    print(f"Warning: Certificate file not found at {cert_path}")
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_CERT_FILE'] = ''
    os.environ['SSL_CERT_DIR'] = ''

# Python HTTPS 검증 완전 비활성화 (개발 환경용)
os.environ['PYTHONHTTPSVERIFY'] = '0'

# .env 파일에서 환경변수 로드 (API 키, 데이터베이스 설정 등)
load_dotenv()  # ANTHROPIC_API_KEY 등의 민감한 정보를 .env 파일에서 읽어옴

# ================================================================================================
# 애플리케이션 설정 상수들
# ================================================================================================

# AI 모델 관련 경로 설정
LOCAL_MODEL_PATH = os.path.abspath("./model_files")  # 로컬 AI 모델이 저장된 절대 경로

# 문서 처리 관련 설정
CHUNK_SIZE = 500      # 문서를 나눌 때 한 청크의 최대 문자 수 (500자 단위로 분할)
CHUNK_OVERLAP = 50    # 청크 간 겹치는 문자 수 (연결성 유지를 위해 50자씩 겹침)
SEARCH_K = 3          # 벡터 검색 시 반환할 유사 문서의 개수 (가장 관련성 높은 3개)

# 데이터베이스 및 캐시 디렉토리
CHROMA_DB_DIR = "./chroma_db"      # ChromaDB 벡터 데이터베이스 저장 경로
MODEL_CACHE_DIR = "./model_cache"  # AI 모델 캐시 저장 경로  
ONNX_MODEL_DIR = "./onnx_models"   # ONNX 변환된 모델 저장 경로

# 세션 관리 설정
SESSION_TIMEOUT = timedelta(hours=1)  # 사용자 세션 만료 시간 (1시간)

# GPU 가속 관련 설정
FORCE_ONNX_MODE = os.getenv("FORCE_ONNX_MODE", "true").lower() == "true"  # ONNX 강제 사용 플래그
EMBEDDING_MODEL = LOCAL_MODEL_PATH  # 임베딩 모델 경로 (로컬 모델 사용)

# 동시 처리 제한 설정 (GPU 메모리 관리용)
MAX_CONCURRENT_GPU_REQUESTS = 5  # 동시에 GPU를 사용할 수 있는 최대 요청 수
BATCH_SIZE = 16                  # 한 번에 처리할 텍스트 배치 크기
gpu_semaphore = Semaphore(MAX_CONCURRENT_GPU_REQUESTS)  # GPU 사용량 제한을 위한 세마포어

# ================================================================================================
# FastAPI 애플리케이션 설정
# ================================================================================================

# FastAPI 앱 인스턴스 생성 (웹 API 서버)
app = FastAPI()

# CORS (Cross-Origin Resource Sharing) 미들웨어 추가
# 웹 브라우저에서 다른 도메인의 API를 호출할 수 있도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 모든 도메인에서 접근 허용 (개발용)
    allow_credentials=True,     # 쿠키 및 인증 정보 허용
    allow_methods=["*"],        # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],        # 모든 HTTP 헤더 허용
)

# ================================================================================================
# 데이터 모델 정의 (Pydantic Models)
# ================================================================================================

class LoadDocumentRequest(BaseModel):
    """
    문서 로드 요청 데이터 모델
    - area: 검토할 영역 (자율/자치활동, 진로활동 등)
    - academic_level: 학업 수준 (고등학교 등)
    """
    area: str            # 특기사항 영역 (예: "자율/자치활동 특기사항")
    academic_level: str  # 학업 수준 (예: "고등학교")

class ReviewRequest(BaseModel):
    """
    문장 검토 요청 데이터 모델
    - statement: 검토할 문장 내용
    - session_id: 세션 식별자 (문서가 로드된 세션)
    """
    statement: str   # 검토 받을 생기부 문장
    session_id: str  # 세션 고유 ID

class ReviewResponse(BaseModel):
    """
    문장 검토 응답 데이터 모델
    - evaluation: 적합성 평가 결과
    - feedback: 검토 의견 및 피드백
    - suggestion: 개선 제안 문장
    - suggestion_length: 개선 제안 문장의 길이
    """
    evaluation: str         # 문장 적합성 평가
    feedback: str          # 상세 피드백
    suggestion: str        # 개선된 문장 제안
    suggestion_length: int # 제안 문장 글자 수

class SessionInfo(BaseModel):
    """
    세션 정보 데이터 모델
    - session_id: 세션 고유 식별자
    - created_at: 세션 생성 시간
    - area: 검토 영역
    - academic_level: 학업 수준
    """
    session_id: str        # 세션 고유 ID
    created_at: datetime   # 세션 생성 시간
    area: str             # 특기사항 영역
    academic_level: str   # 학업 수준

# ================================================================================================
# 전역 변수 선언
# ================================================================================================

# AI 모델 관련 전역 변수
tokenizer = None     # 텍스트를 토큰으로 변환하는 토크나이저 객체
ort_session = None   # ONNX Runtime 세션 객체 (GPU 가속 모델 추론용)

# 세션 관리용 딕셔너리 (메모리 상에서 사용자 세션 정보 저장)
sessions: Dict[str, dict] = {}  # 키: session_id, 값: 세션 데이터 딕셔너리

# ================================================================================================
# GPU 관련 유틸리티 함수들
# ================================================================================================

def check_cuda_availability():
    """
    CUDA GPU 사용 가능 여부를 확인하고 GPU 정보를 반환하는 함수
    
    Returns:
        tuple: (cuda_available, gpu_name, total_memory)
        - cuda_available (bool): CUDA 사용 가능 여부
        - gpu_name (str): GPU 이름 (사용 불가시 None)
        - total_memory (float): GPU 메모리 크기 (GB 단위, 알 수 없으면 "알 수 없음")
    """
    # ONNX Runtime이 설치되지 않은 경우 CUDA 사용 불가
    if not ONNX_AVAILABLE:
        return False, None, None
        
    try:
        # ONNX Runtime에서 사용 가능한 실행 프로바이더 목록 조회
        providers = ort.get_available_providers()
        cuda_available = 'CUDAExecutionProvider' in providers  # CUDA 프로바이더 존재 여부 확인
        
        if cuda_available:
            # CUDA가 사용 가능한 경우 GPU 정보 수집 시도
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()  # NVIDIA Management Library 초기화
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 첫 번째 GPU 핸들 획득
                    gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')  # GPU 이름 조회
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 메모리 정보 조회
                    total_memory = int(memory_info.total) / (1024**3)  # 바이트를 GB로 변환
                    return True, gpu_name, total_memory
                except Exception as e:
                    print(f"pynvml 오류: {str(e)}")
                    return True, "NVIDIA GPU", "알 수 없음"  # GPU는 있지만 정보 조회 실패
            else:
                # pynvml이 없는 경우 기본 정보만 반환
                return True, "NVIDIA GPU", "알 수 없음"
        else:
            # CUDA 프로바이더가 없는 경우
            return False, None, None
    except Exception as e:
        print(f"CUDA 확인 중 오류: {str(e)}")
        return False, None, None

def setup_onnx_providers():
    """
    ONNX Runtime 실행 프로바이더를 설정하는 함수
    GPU가 사용 가능하면 CUDA 프로바이더를, 아니면 CPU 프로바이더를 설정
    
    Returns:
        list: ONNX Runtime 프로바이더 설정 리스트
    """
    # ONNX Runtime이 설치되지 않은 경우 예외 발생
    if not ONNX_AVAILABLE:
        raise Exception("ONNX Runtime이 설치되지 않았습니다.")
        
    # GPU 사용 가능 여부 확인
    cuda_available, gpu_name, gpu_memory = check_cuda_availability()
    
    if cuda_available:
        # GPU 사용 가능한 경우의 로그 출력
        print(f"🚀 CUDA GPU 사용: {gpu_name}")
        if gpu_memory != "알 수 없음":
            print(f"📊 GPU 메모리: {gpu_memory:.1f}GB")
        print(f"👥 최대 동시 사용자: {MAX_CONCURRENT_GPU_REQUESTS}")
        print(f"📦 배치 처리 크기: {BATCH_SIZE}")
        
        # CUDA 프로바이더 설정 (GPU 가속 활성화)
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,  # 사용할 GPU 장치 ID (첫 번째 GPU)
                'arena_extend_strategy': 'kNextPowerOfTwo',  # 메모리 할당 전략
                'gpu_mem_limit': int(2 * 1024 * 1024 * 1024),  # GPU 메모리 제한 (2GB)
                'cudnn_conv_algo_search': 'EXHAUSTIVE',  # 최적의 CNN 알고리즘 탐색
                'do_copy_in_default_stream': True,  # 기본 스트림에서 메모리 복사
            }),
            'CPUExecutionProvider'  # CUDA 실패시 CPU로 폴백
        ]
    else:
        # GPU를 사용할 수 없는 경우 CPU 모드로 실행
        print("🖥️ CPU 모드로 실행합니다.")
        providers = ['CPUExecutionProvider']  # CPU 프로바이더만 사용
    
    return providers

# ================================================================================================
# AI 모델 로드 및 ONNX 변환 함수
# ================================================================================================

def download_and_cache_model():
    """
    로컬 AI 모델을 ONNX 형식으로 변환하고 GPU 가속을 설정하는 함수
    
    이 함수는 다음 작업을 수행합니다:
    1. 로컬에 저장된 Transformers 모델 확인
    2. 필요한 경우 ONNX 형식으로 변환 (GPU 최적화)
    3. 토크나이저와 ONNX 세션 초기화
    4. 전역 변수에 모델 객체들 저장
    
    주의: SentenceTransformer를 직접 사용하지 않고 Transformers + ONNX 조합 사용
    """
    global tokenizer, ort_session  # 전역 변수 수정을 위한 선언
    
    # 1단계: 로컬 모델 파일 존재 여부 확인
    if not os.path.exists(EMBEDDING_MODEL):
        raise HTTPException(status_code=500, detail=f"로컬 모델을 찾을 수 없습니다: {EMBEDDING_MODEL}")
    
    try:
        print("🔄 로컬 모델 직접 ONNX 변환 중...")
        print(f"📍 모델 경로: {EMBEDDING_MODEL}")
        
        # 2단계: 필수 패키지 설치 확인
        if not ONNX_AVAILABLE or not OPTIMUM_AVAILABLE:
            raise Exception("ONNX Runtime 또는 Optimum 패키지가 설치되지 않았습니다.")
        
        # 3단계: 필요한 디렉토리 생성
        for dir_path in [MODEL_CACHE_DIR, ONNX_MODEL_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)  # 캐시 및 ONNX 모델 저장용 디렉토리 생성
        
        # 4단계: 토크나이저 로드 (텍스트를 숫자로 변환하는 도구)
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, local_files_only=True)
        
        # 5단계: ONNX 모델 파일 경로 설정
        onnx_model_path = os.path.join(ONNX_MODEL_DIR, "model.onnx")
        
        # 6단계: ONNX 모델이 없으면 새로 변환, 있으면 기존 파일 사용
        if not os.path.exists(onnx_model_path):
            print("📦 Transformers 모델을 직접 ONNX로 변환 중...")
            
            # *** 핵심: SentenceTransformer 대신 직접 변환 방식 ***
            from transformers.models.auto.modeling_auto import AutoModel
            
            # 6-1. PyTorch 형태의 원본 모델 로드
            pytorch_model = AutoModel.from_pretrained(EMBEDDING_MODEL, local_files_only=True)
            
            # 6-2. ONNX 변환을 위한 임시 디렉토리 생성
            temp_model_dir = os.path.join(MODEL_CACHE_DIR, "temp_for_onnx")
            os.makedirs(temp_model_dir, exist_ok=True)
            
            # 6-3. 임시 디렉토리에 모델과 토크나이저 저장
            pytorch_model.save_pretrained(temp_model_dir)  # 모델 저장
            tokenizer.save_pretrained(temp_model_dir)      # 토크나이저 저장
            
            # 6-4. ONNX 변환 실행 (GPU 최적화된 형태로 변환)
            model = ORTModelForFeatureExtraction.from_pretrained(
                temp_model_dir,      # 변환할 모델 경로
                export=True,         # ONNX로 변환 활성화
                local_files_only=True # 로컬 파일만 사용
            )
            model.save_pretrained(ONNX_MODEL_DIR)  # 변환된 ONNX 모델 저장
            
            # 6-5. 임시 디렉토리 정리 (공간 절약)
            import shutil
            shutil.rmtree(temp_model_dir)
            
        else:
            # 이미 ONNX 변환된 모델이 있는 경우 로드
            print("📁 기존 ONNX 모델 로드 중...")
            model = ORTModelForFeatureExtraction.from_pretrained(ONNX_MODEL_DIR)
        
        # 7단계: ONNX Runtime 세션 생성 (실제 추론 엔진)
        providers = setup_onnx_providers()  # GPU/CPU 프로바이더 설정
        
        # 7-1. ONNX 파일 경로 찾기
        onnx_files = [f for f in os.listdir(ONNX_MODEL_DIR) if f.endswith('.onnx')]
        if onnx_files:
            actual_onnx_path = os.path.join(ONNX_MODEL_DIR, onnx_files[0])  # 첫 번째 .onnx 파일 사용
        else:
            actual_onnx_path = model.model_path  # 모델 객체에서 경로 가져오기
        
        # 7-2. ONNX Runtime 세션 옵션 설정
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # 모든 최적화 활성화
        
        # 7-3. ONNX Runtime 추론 세션 생성
        ort_session = ort.InferenceSession(
            actual_onnx_path,  # ONNX 모델 파일 경로
            sess_options,      # 세션 옵션 (최적화 설정)
            providers=providers # 실행 프로바이더 (GPU/CPU)
        )
        
        # 성공 메시지 출력
        print("✅ 로컬 모델 ONNX+GPU 변환 완료!")
        print(f"🔧 사용 중인 프로바이더: {ort_session.get_providers()}")
        
        return tokenizer, ort_session  # 토크나이저와 ONNX 세션 반환
        
    except Exception as e:
        # 오류 발생시 상세 로그 출력 및 HTTP 예외 발생
        print(f"모델 로드 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"직접 ONNX 변환 중 오류: {str(e)}")

# ================================================================================================
# 임베딩 생성 함수들 (텍스트를 벡터로 변환)
# ================================================================================================

async def get_embeddings_batch_unified(texts, tokenizer, model_or_session):
    """
    여러 텍스트를 한 번에 벡터로 변환하는 통합 함수
    PyTorch 모델과 ONNX Runtime 세션 모두 지원
    
    Args:
        texts (list): 변환할 텍스트 리스트
        tokenizer: 토크나이저 객체 (텍스트→토큰 변환)
        model_or_session: PyTorch 모델 또는 ONNX Runtime 세션
        
    Returns:
        list: 각 텍스트에 대응하는 임베딩 벡터 리스트
    """
    # GPU 사용량 제한을 위한 세마포어 사용 (동시 처리 제한)
    async with gpu_semaphore:
        try:
            all_embeddings = []  # 모든 임베딩 결과를 저장할 리스트
            
            # 모델 타입 확인 (ONNX Runtime 세션인지 PyTorch 모델인지)
            is_onnx = hasattr(model_or_session, 'run')  # ONNX 세션은 run 메서드가 있음
            
            # 배치 단위로 처리 (메모리 효율성을 위해)
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]  # 현재 배치의 텍스트들
                
                if is_onnx:
                    # ONNX Runtime을 사용한 추론 방식
                    
                    # 1. 토크나이저로 텍스트를 숫자 배열로 변환
                    inputs = tokenizer(
                        batch_texts,           # 처리할 텍스트 배치
                        padding=True,          # 배치 내 최대 길이로 패딩
                        truncation=True,       # 최대 길이 초과시 자르기
                        max_length=512,        # 최대 토큰 길이 (BERT 계열 표준)
                        return_tensors="np"    # NumPy 배열로 반환
                    )
                    
                    # 2. ONNX 입력 형식으로 변환 (int64 타입 필수)
                    ort_inputs = {
                        'input_ids': inputs['input_ids'].astype(np.int64),         # 토큰 ID 배열
                        'attention_mask': inputs['attention_mask'].astype(np.int64) # 패딩 마스크
                    }
                    
                    # 3. ONNX 모델로 추론 실행
                    outputs = model_or_session.run(None, ort_inputs)  # 순전파 실행
                    last_hidden_state = outputs[0]  # 마지막 은닉층 출력 (문맥 임베딩)
                    
                    # 4. Mean Pooling 수행 (토큰 임베딩들의 평균 계산)
                    attention_mask = inputs['attention_mask']
                    mask_expanded = np.expand_dims(attention_mask, axis=-1)  # 차원 확장
                    mask_expanded = np.broadcast_to(mask_expanded, last_hidden_state.shape)  # 브로드캐스팅
                    
                    # 패딩 토큰은 제외하고 평균 계산
                    sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)  # 마스킹 후 합계
                    sum_mask = np.sum(mask_expanded, axis=1)  # 유효 토큰 개수
                    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)  # 0으로 나누기 방지
                    
                    embeddings = sum_embeddings / sum_mask  # 평균 임베딩 계산
                    
                    # 5. L2 정규화 (벡터 길이를 1로 만들어 코사인 유사도 계산 최적화)
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)  # 벡터 크기 계산
                    norms = np.clip(norms, a_min=1e-9, a_max=None)  # 0으로 나누기 방지
                    embeddings = embeddings / norms  # 정규화된 임베딩
                    
                    batch_embeddings = embeddings.tolist()  # Python 리스트로 변환
                    
                else:
                    # PyTorch를 사용한 추론 방식
                    import torch
                    
                    # 1. 모델이 실행되고 있는 디바이스 확인 (CPU or GPU)
                    device = next(model_or_session.parameters()).device
                    
                    # 2. 토크나이저로 텍스트를 PyTorch 텐서로 변환
                    inputs = tokenizer(
                        batch_texts,           # 처리할 텍스트 배치
                        padding=True,          # 배치 내 최대 길이로 패딩
                        truncation=True,       # 최대 길이 초과시 자르기
                        max_length=512,        # 최대 토큰 길이
                        return_tensors="pt"    # PyTorch 텐서로 반환
                    )
                    
                    # 3. 입력 텐서를 모델과 같은 디바이스로 이동 (GPU/CPU)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 4. 그래디언트 계산 비활성화 (추론 모드, 메모리 절약)
                    with torch.no_grad():
                        outputs = model_or_session(**inputs)  # 모델 순전파
                        last_hidden_state = outputs.last_hidden_state  # 마지막 은닉층 출력
                        
                        # 5. Mean Pooling 수행 (토큰 임베딩들의 평균 계산)
                        attention_mask = inputs['attention_mask']
                        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                        
                        # 패딩 토큰 제외하고 평균 계산
                        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)  # 마스킹 후 합계
                        sum_mask = torch.sum(mask_expanded, 1)  # 유효 토큰 개수
                        sum_mask = torch.clamp(sum_mask, min=1e-9)  # 0으로 나누기 방지
                        
                        embeddings = sum_embeddings / sum_mask  # 평균 임베딩 계산
                        
                        # 6. L2 정규화 (벡터 길이를 1로 만들기)
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        
                        # 7. CPU로 이동 후 Python 리스트로 변환
                        batch_embeddings = embeddings.cpu().tolist()
                
                # 현재 배치의 임베딩을 전체 결과에 추가
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings  # 모든 텍스트의 임베딩 벡터 반환
            
        except Exception as e:
            # 임베딩 생성 중 오류 발생시 HTTP 예외로 변환
            raise HTTPException(status_code=500, detail=f"임베딩 생성 중 오류: {str(e)}")

async def get_embeddings_unified(text, tokenizer, model_or_session):
    """
    단일 텍스트를 벡터로 변환하는 함수
    내부적으로 배치 처리 함수를 호출하여 일관성 유지
    
    Args:
        text (str): 변환할 단일 텍스트
        tokenizer: 토크나이저 객체
        model_or_session: PyTorch 모델 또는 ONNX Runtime 세션
        
    Returns:
        list: 단일 텍스트의 임베딩 벡터
    """
    # 단일 텍스트를 리스트로 감싸서 배치 함수 호출
    results = await get_embeddings_batch_unified([text], tokenizer, model_or_session)
    return results[0]  # 첫 번째 (유일한) 결과 반환

class UnifiedEmbeddingFunction(Embeddings):
    """
    ChromaDB 호환 임베딩 함수 클래스
    PyTorch와 ONNX Runtime을 모두 지원하는 통합 임베딩 인터페이스
    
    LangChain의 Embeddings 인터페이스를 상속받아 ChromaDB에서 사용 가능
    """
    def __init__(self, tokenizer, model_or_session):
        """
        임베딩 함수 초기화
        
        Args:
            tokenizer: 토크나이저 객체
            model_or_session: PyTorch 모델 또는 ONNX Runtime 세션
        """
        self.tokenizer = tokenizer           # 텍스트→토큰 변환기
        self.model_or_session = model_or_session  # AI 모델 객체

    def embed_documents(self, texts):
        """
        여러 문서를 벡터로 변환하는 메서드 (ChromaDB 호환)
        ChromaDB가 문서들을 벡터 데이터베이스에 저장할 때 호출
        
        Args:
            texts (list): 변환할 문서 텍스트 리스트
            
        Returns:
            list: 각 문서의 임베딩 벡터 리스트
        """
        try:
            return self._get_embeddings_batch_sync(texts)  # 동기 방식 배치 처리
        except Exception as e:
            print(f"문서 임베딩 오류: {str(e)}")
            raise

    def embed_query(self, text):
        """
        단일 검색 쿼리를 벡터로 변환하는 메서드 (ChromaDB 호환)
        사용자 질문을 벡터로 변환하여 유사한 문서 검색에 사용
        
        Args:
            text (str): 검색 쿼리 텍스트
            
        Returns:
            list: 쿼리의 임베딩 벡터
        """
        try:
            results = self._get_embeddings_batch_sync([text])  # 단일 텍스트를 배치로 처리
            return results[0]  # 첫 번째 결과 반환
        except Exception as e:
            print(f"쿼리 임베딩 오류: {str(e)}")
            raise

    def _get_embeddings_batch_sync(self, texts):
        """
        동기 방식으로 여러 텍스트를 배치 처리하여 임베딩 생성
        ChromaDB에서 호출하는 메서드 (비동기 함수를 동기로 래핑)
        
        Args:
            texts (list): 변환할 텍스트 리스트
            
        Returns:
            list: 각 텍스트의 임베딩 벡터 리스트
        """
        try:
            all_embeddings = []  # 모든 임베딩 결과 저장용
            
            # 모델 타입 확인 (ONNX Runtime 세션인지 PyTorch 모델인지)
            is_onnx = hasattr(self.model_or_session, 'run')  # ONNX 세션은 run 메서드 있음
            
            # 배치 단위로 처리 (메모리 효율성)
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]  # 현재 배치 텍스트들
                
                if is_onnx:
                    # ONNX Runtime을 사용한 동기 방식 추론
                    
                    # 1. 토크나이저로 텍스트를 NumPy 배열로 변환
                    inputs = self.tokenizer(
                        batch_texts,           # 처리할 텍스트 배치
                        padding=True,          # 배치 내 최대 길이로 패딩
                        truncation=True,       # 최대 길이 초과시 자르기
                        max_length=512,        # 최대 토큰 길이
                        return_tensors="np"    # NumPy 배열로 반환
                    )
                    
                    # 2. ONNX 입력 형식으로 변환
                    ort_inputs = {
                        'input_ids': inputs['input_ids'].astype(np.int64),         # 토큰 ID
                        'attention_mask': inputs['attention_mask'].astype(np.int64) # 패딩 마스크
                    }
                    
                    # 3. ONNX 모델 추론 실행
                    outputs = self.model_or_session.run(None, ort_inputs)
                    last_hidden_state = outputs[0]  # 마지막 은닉층 출력
                    
                    # 4. Mean Pooling 수행
                    attention_mask = inputs['attention_mask']
                    mask_expanded = np.expand_dims(attention_mask, axis=-1)
                    mask_expanded = np.broadcast_to(mask_expanded, last_hidden_state.shape)
                    
                    # 패딩 토큰 제외하고 평균 계산
                    sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
                    sum_mask = np.sum(mask_expanded, axis=1)
                    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)  # 0으로 나누기 방지
                    
                    embeddings = sum_embeddings / sum_mask  # 평균 임베딩
                    
                    # 5. L2 정규화
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms = np.clip(norms, a_min=1e-9, a_max=None)
                    embeddings = embeddings / norms  # 정규화된 임베딩
                    
                    batch_embeddings = embeddings.tolist()  # Python 리스트로 변환
                    
                else:
                    # PyTorch를 사용한 동기 방식 추론
                    import torch
                    
                    # 1. 모델 디바이스 확인
                    device = next(self.model_or_session.parameters()).device
                    
                    # 2. 토크나이저로 PyTorch 텐서 생성
                    inputs = self.tokenizer(
                        batch_texts,           # 처리할 텍스트 배치
                        padding=True,          # 패딩 적용
                        truncation=True,       # 길이 제한
                        max_length=512,        # 최대 토큰 길이
                        return_tensors="pt"    # PyTorch 텐서로 반환
                    )
                    
                    # 3. 입력을 모델과 같은 디바이스로 이동
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # 4. 그래디언트 비활성화하여 추론 수행
                    with torch.no_grad():
                        outputs = self.model_or_session(**inputs)  # 모델 추론
                        last_hidden_state = outputs.last_hidden_state
                        
                        # 5. Mean Pooling 수행
                        attention_mask = inputs['attention_mask']
                        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                        
                        # 패딩 토큰 제외하고 평균 계산
                        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                        sum_mask = torch.sum(mask_expanded, 1)
                        sum_mask = torch.clamp(sum_mask, min=1e-9)  # 0으로 나누기 방지
                        
                        embeddings = sum_embeddings / sum_mask  # 평균 임베딩
                        
                        # 6. L2 정규화
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        
                        # 7. CPU로 이동 후 리스트 변환
                        batch_embeddings = embeddings.cpu().tolist()
                
                # 배치 결과를 전체 결과에 추가
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings  # 모든 임베딩 반환
            
        except Exception as e:
            # 오류 발생시 예외 전파
            raise Exception(f"임베딩 생성 중 오류: {str(e)}")

# ================================================================================================
# 세션 관리 함수들
# ================================================================================================

def cleanup_expired_sessions():
    """
    만료된 사용자 세션들을 정리하는 함수
    메모리 누수 방지 및 디스크 공간 절약을 위해 정기적으로 호출됨
    
    처리 과정:
    1. 현재 시간과 비교하여 만료된 세션 식별
    2. 메모리에서 벡터스토어 객체 정리
    3. 파일 시스템에서 세션 디렉토리 삭제
    4. 세션 딕셔너리에서 제거
    """
    current_time = datetime.now()  # 현재 시간 획득
    
    # 만료된 세션 ID 목록 생성 (리스트 컴프리헨션 사용)
    expired_sessions = [
        session_id for session_id, session_data in sessions.items()
        if current_time - session_data['created_at'] > SESSION_TIMEOUT  # 1시간 초과 세션
    ]
    
    # 각 만료된 세션에 대해 정리 작업 수행
    for session_id in expired_sessions:
        try:
            # 1단계: 메모리에서 벡터스토어 객체 정리
            if session_id in sessions and 'vectorstore' in sessions[session_id]:
                vectorstore = sessions[session_id]['vectorstore']  # 벡터스토어 객체 획득
                # ChromaDB 클라이언트 연결 정리
                if hasattr(vectorstore, '_client') and vectorstore._client:
                    try:
                        vectorstore._client.reset()  # 클라이언트 리셋
                    except:
                        pass  # 리셋 실패해도 계속 진행
                del vectorstore  # 객체 삭제로 메모리 해제
            
            # 2단계: 세션 딕셔너리에서 제거
            if session_id in sessions:
                del sessions[session_id]  # 메모리에서 세션 데이터 제거
            
            # 3단계: 파일 시스템에서 세션 디렉토리 삭제 (재시도 메커니즘 포함)
            session_dir = os.path.join(CHROMA_DB_DIR, session_id)  # 세션 디렉토리 경로
            if os.path.exists(session_dir):
                import shutil  # 디렉토리 삭제용
                import time    # 대기 시간용
                
                # Windows 파일 시스템의 파일 잠금 문제 해결을 위한 재시도 로직
                max_retries = 3  # 최대 3번 재시도
                for attempt in range(max_retries):
                    try:
                        # 메모리 정리로 파일 핸들 해제
                        import gc
                        gc.collect()  # 가비지 컬렉션 강제 실행
                        
                        # 첫 번째 시도가 아니면 잠시 대기 (파일 잠금 해제 대기)
                        if attempt > 0:
                            time.sleep(0.5)  # 0.5초 대기
                        
                        shutil.rmtree(session_dir)  # 디렉토리 및 하위 파일 모두 삭제
                        print(f"✅ 만료된 세션 디렉토리 삭제 완료: {session_id}")
                        break  # 성공시 재시도 루프 종료
                        
                    except PermissionError as e:
                        # 파일 권한 오류 (Windows에서 흔히 발생)
                        if attempt == max_retries - 1:
                            print(f"⚠️ 세션 디렉토리 삭제 실패 (권한 문제): {session_id} - {str(e)}")
                            # 삭제 실패해도 메모리는 이미 정리했으므로 계속 진행
                        else:
                            print(f"🔄 세션 디렉토리 삭제 재시도 중... ({attempt + 1}/{max_retries}): {session_id}")
                    except Exception as e:
                        print(f"❌ 세션 디렉토리 삭제 중 예외 발생: {session_id} - {str(e)}")
                        break  # 다른 예외 발생시 재시도 중단
                        
        except Exception as e:
            print(f"❌ 세션 정리 중 예외 발생: {session_id} - {str(e)}")
            # 개별 세션 정리 실패해도 다른 세션들은 계속 처리 (continue로 다음 세션으로)
            continue

# ================================================================================================
# REST API 엔드포인트들
# ================================================================================================

@app.post("/api/load-documents")
async def load_documents(request: LoadDocumentRequest):
    """
    문서 로드 API 엔드포인트
    
    기능:
    1. 선택된 영역(자율/자치활동 또는 진로활동)의 가이드라인 문서들을 로드
    2. 문서들을 청크 단위로 분할하여 벡터 데이터베이스에 저장
    3. 사용자별 고유 세션 생성 및 관리
    
    Args:
        request (LoadDocumentRequest): 로드할 영역과 학업 수준 정보
        
    Returns:
        dict: 세션 ID와 서버 정보가 포함된 응답
    """
    try:
        # 1단계: 만료된 기존 세션들을 정리하여 메모리 확보
        cleanup_expired_sessions()
        
        # 2단계: 사용자가 선택한 영역을 실제 디렉토리명으로 매핑
        area_map = {
            "자율/자치활동 특기사항": "self_governance_guidelines",  # 자율/자치활동 가이드라인
            "진로활동 특기사항": "career_activity_guidelines"        # 진로활동 가이드라인
        }
        
        # 3단계: 유효한 영역인지 검증
        if request.area not in area_map:
            raise HTTPException(status_code=400, detail=f"Invalid area selected: {request.area}")
        
        # 4단계: 해당 영역의 문서 디렉토리 경로 생성
        directory = f"data/{area_map[request.area]}"
        if not os.path.exists(directory):
            raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
        
        # 5단계: 디렉토리에 문서 파일들이 있는지 확인
        if not os.listdir(directory):
            raise HTTPException(status_code=404, detail=f"Directory is empty: {directory}")
        
        # 6단계: 새로운 사용자 세션을 위한 고유 ID 생성
        session_id = str(uuid.uuid4())  # UUID4로 충돌 없는 고유 ID 생성
        session_db_dir = os.path.join(CHROMA_DB_DIR, session_id)  # 세션별 데이터베이스 디렉토리
        
        # 7단계: 디렉토리에서 마크다운 문서 파일들을 로드
        documents = []  # 로드된 문서들을 저장할 리스트
        for file_path in os.listdir(directory):
            try:
                if file_path.endswith('.md'):  # 마크다운 파일만 처리
                    with open(os.path.join(directory, file_path), 'r', encoding='utf-8') as f:
                        content = f.read()  # 파일 내용 읽기
                        # Document 객체 생성 (내용 + 메타데이터)
                        documents.append(Document(page_content=content, metadata={"source": file_path}))
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")  # 파일 읽기 오류 로그
        
        # 8단계: 로드된 문서가 있는지 확인
        if len(documents) == 0:
            raise HTTPException(status_code=404, detail=f"No markdown files found in {directory}")
        
        # 9단계: 문서들을 검색 가능한 작은 청크로 분할
        text_splitter = MarkdownTextSplitter(
            chunk_size=CHUNK_SIZE,      # 청크당 최대 문자 수 (500자)
            chunk_overlap=CHUNK_OVERLAP # 청크 간 겹치는 문자 수 (50자, 문맥 연결성 유지)
        )
        splits = text_splitter.split_documents(documents)  # 문서 분할 실행
        
        # 10단계: AI 모델이 초기화되지 않았으면 로드
        if tokenizer is None or ort_session is None:
            try:
                download_and_cache_model()  # 모델 다운로드 및 ONNX 변환
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")
        
        # 11단계: 임베딩 함수 생성 (ChromaDB 호환 인터페이스)
        embedding_function = UnifiedEmbeddingFunction(tokenizer, ort_session)
        
        try:
            # 12단계: 세션별 벡터 데이터베이스 디렉토리 생성
            os.makedirs(session_db_dir, exist_ok=True)
            
            # 13단계: 사용 중인 모델 타입 확인 (로그 출력용)
            is_local_model = (os.path.exists(EMBEDDING_MODEL) and 
                            os.path.isdir(EMBEDDING_MODEL) and
                            os.path.exists(os.path.join(EMBEDDING_MODEL, "config.json")))
            
            # 14단계: 현재 사용 중인 가속 방식 결정
            if FORCE_ONNX_MODE:
                acceleration_type = "ONNX Runtime (GPU)"  # 강제 ONNX 모드
            elif hasattr(ort_session, 'run'):  # ONNX 세션 확인
                acceleration_type = "ONNX Runtime"
            else:  # PyTorch 모델
                acceleration_type = "PyTorch (로컬)"
            
            # 15단계: 문서 처리 시작 로그
            print(f"📊 {acceleration_type}로 문서 {len(splits)}개 처리 중... (사용자: {len(sessions)+1}명)")
            start_time = time.time()  # 처리 시간 측정 시작
            
            # 16단계: ChromaDB 벡터 데이터베이스 생성 (핵심 과정)
            vectorstore = Chroma.from_documents(
                documents=splits,               # 분할된 문서 청크들
                embedding=embedding_function,   # 임베딩 함수 (텍스트→벡터 변환)
                persist_directory=session_db_dir # 데이터베이스 저장 경로
            )
            
            end_time = time.time()  # 처리 시간 측정 완료
            print(f"⚡ {acceleration_type} 문서 처리 완료: {end_time - start_time:.1f}초")
            
            # 17단계: 세션 정보를 메모리에 저장
            sessions[session_id] = {
                'vectorstore': vectorstore,     # 벡터 데이터베이스 객체
                'created_at': datetime.now(),   # 세션 생성 시간
                'area': request.area,           # 선택된 영역
                'academic_level': request.academic_level  # 학업 수준
            }
            
        except Exception as e:
            # 벡터스토어 생성 중 오류 발생시 HTTP 예외로 변환
            raise HTTPException(status_code=500, detail=f"Error creating vectorstore: {str(e)}")
        
        # 18단계: 메모리 정리 (가비지 컬렉션)
        gc.collect()
        
        # 19단계: 현재 활성 세션 수 확인
        active_sessions = len(sessions)
        
        # 20단계: 성공 응답 반환
        return {
            "status": "success",
            "message": f"Documents loaded successfully with {acceleration_type}",
            "session_id": session_id,  # 클라이언트가 후속 요청에서 사용할 세션 ID
            "server_info": {
                "active_sessions": active_sessions,  # 현재 활성 사용자 수
                "processing_time": f"{end_time - start_time:.1f}s",  # 처리 소요 시간
                "acceleration": acceleration_type,   # 사용된 가속 방식
                "model_type": "로컬 모델" if is_local_model else "온라인 모델"  # 모델 타입
            }
        }

    except Exception as e:
        # 예외 발생시 상세한 디버깅 정보 출력
        import traceback
        print("--- LOAD DOCUMENTS ENDPOINT ERROR ---")
        traceback.print_exc()  # 전체 스택 트레이스 출력
        print("-------------------------------------")
        # 클라이언트에게 일반적인 오류 메시지 반환
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/api/review")
async def review_statement(request: ReviewRequest):
    """
    생기부 문장 검토 API 엔드포인트
    
    기능:
    1. 사용자가 작성한 생기부 문장을 AI로 분석
    2. 적합성 평가, 개선 의견, 수정 제안을 제공
    3. 세션별 벡터 데이터베이스에서 관련 가이드라인 검색
    4. Claude AI를 활용한 전문적인 피드백 생성
    
    Args:
        request (ReviewRequest): 검토할 문장과 세션 ID
        
    Returns:
        ReviewResponse: 평가, 피드백, 개선 제안이 포함된 응답
    """
    # 1단계: 만료된 세션 정리 (메모리 관리)
    cleanup_expired_sessions()
    
    # 2단계: 세션 ID 유효성 검사
    if request.session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid or expired session ID")
    
    # 3단계: 세션 데이터 및 벡터스토어 획득
    session_data = sessions[request.session_id]
    vectorstore = session_data['vectorstore']  # 해당 세션의 벡터 데이터베이스
    
    # 4단계: 입력 문장 유효성 검사
    if not request.statement:
        raise HTTPException(status_code=400, detail="Statement is required")
    
    try:
        # 5단계: Claude AI에게 보낼 프롬프트 템플릿 정의
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

        # 6단계: 디버깅을 위한 로그 출력
        print("=== REVIEW API DEBUG ===")
        print(f"Session ID: {request.session_id}")
        print(f"Statement: {request.statement}")
        print(f"Vectorstore type: {type(vectorstore)}")
        
        # 7단계: AI 체인 생성 (벡터 검색 + Claude AI 조합)
        try:
            chain = create_chain(vectorstore)  # 벡터스토어와 Claude를 연결하는 체인 생성
            print("Chain created successfully")
        except Exception as chain_error:
            print(f"Chain creation error: {str(chain_error)}")
            raise HTTPException(status_code=500, detail=f"Chain creation failed: {str(chain_error)}")
        
        # 8단계: Claude API 호출 전 프롬프트 확인 (디버깅용)
        print("Claude API 호출 프롬프트 전체:")
        print(template.replace("{question}", request.statement))
        
        # 9단계: AI 체인을 통해 문장 분석 실행
        try:
            response = chain.invoke(request.statement)  # 사용자 문장을 AI에게 전달
            print("Chain invoke completed")
        except Exception as invoke_error:
            print(f"Chain invoke error: {str(invoke_error)}")
            raise HTTPException(status_code=500, detail=f"Chain invoke failed: {str(invoke_error)}")
        
        # 10단계: Claude 응답에서 실제 내용 추출
        result = getattr(response, "content", None)  # 응답 객체에서 content 속성 추출
        if not result and isinstance(response, dict):
            result = response.get("content", "")  # 딕셔너리 형태인 경우 content 키 확인
        elif not result:
            result = str(response)  # 다른 형태인 경우 문자열로 변환

        print("Claude API 응답 원문:\n", result)

        # 11단계: Claude 응답을 세 가지 항목으로 파싱
        eval_part, feedback_part, suggestion_part = "", "", ""
        try:
            # "①", "②", "③" 기호를 기준으로 문장 분할
            parts = result.split("①")[1].split("②")  # ①을 기준으로 나누고 ②으로 다시 분할
            eval_part = parts[0].strip()  # 적합성 평가 부분
            feedback_suggestion = parts[1].split("③")  # ②와 ③으로 분할
            feedback_part = feedback_suggestion[0].strip()  # 검토 의견 부분
            suggestion_part = feedback_suggestion[1].strip()  # 개선 제안 부분
        except Exception as parse_error:
            # 파싱 실패시 전체 응답을 평가 부분에 저장
            print(f"Parsing error: {str(parse_error)}")
            eval_part = result.strip()

        # 12단계: 각 섹션에서 불필요한 제목 텍스트 제거
        eval_part = remove_heading(eval_part, "적합성 평가")    # "적합성 평가:" 등 제목 제거
        feedback_part = remove_heading(feedback_part, "검토 의견")  # "검토 의견:" 등 제목 제거
        suggestion_part = remove_heading(suggestion_part, "개선 제안")  # "개선 제안:" 등 제목 제거

        # 13단계: 각 섹션의 가독성 향상을 위한 포맷팅
        eval_part = prettify_evaluation(eval_part)      # 평가 섹션 이모지 및 포맷팅
        feedback_part = prettify_feedback(feedback_part)  # 피드백 섹션 이모지 및 포맷팅
        suggestion_part = prettify_suggestion(suggestion_part)  # 제안 섹션 포맷팅

        # 14단계: 개선 제안을 500자로 제한 (생기부 글자 수 제한)
        suggestion_part = suggestion_part[:500]  # 최대 500자까지만 사용
        suggestion_length = len(suggestion_part)  # 실제 글자 수 계산

        # 15단계: 성공 로그 출력 및 응답 반환
        print("=== REVIEW API SUCCESS ===")
        return {
            "evaluation": eval_part,          # 적합성 평가 결과
            "feedback": feedback_part,        # 검토 의견 및 피드백
            "suggestion": suggestion_part,    # 개선된 문장 제안
            "suggestion_length": suggestion_length  # 제안 문장의 글자 수
        }

    except Exception as e:
        # 예외 발생시 상세한 오류 정보 로그 출력
        import traceback
        print("=== REVIEW API ERROR ===")
        print(f"Error: {str(e)}")
        traceback.print_exc()  # 전체 스택 트레이스 출력
        print("========================")
        # 클라이언트에게 오류 메시지 반환
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def list_sessions():
    """
    활성 세션 목록 조회 API 엔드포인트
    
    기능:
    1. 현재 서버에서 관리 중인 모든 활성 세션 정보 반환
    2. 만료된 세션들을 먼저 정리한 후 목록 제공
    3. 각 세션의 생성 시간, 영역, 학업 수준 정보 포함
    
    Returns:
        list[SessionInfo]: 활성 세션 정보 리스트
    """
    cleanup_expired_sessions()  # 만료된 세션 정리 후 목록 반환
    return [
        SessionInfo(
            session_id=session_id,                    # 세션 고유 ID
            created_at=session_data['created_at'],    # 세션 생성 시간
            area=session_data['area'],                # 선택된 영역
            academic_level=session_data['academic_level']  # 학업 수준
        )
        for session_id, session_data in sessions.items()  # 모든 활성 세션 순회
    ]

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    세션 삭제 API 엔드포인트
    
    기능:
    1. 지정된 세션 ID의 세션을 강제로 삭제
    2. 메모리에서 벡터스토어 객체 정리
    3. 파일 시스템에서 세션 디렉토리 삭제
    
    Args:
        session_id (str): 삭제할 세션의 고유 ID
        
    Returns:
        dict: 삭제 성공 메시지
    """
    # 1단계: 세션 존재 여부 확인
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # 2단계: 메모리에서 벡터스토어 객체 정리
        if 'vectorstore' in sessions[session_id]:
            vectorstore = sessions[session_id]['vectorstore']  # 벡터스토어 객체 획득
            # ChromaDB 클라이언트 연결 정리
            if hasattr(vectorstore, '_client') and vectorstore._client:
                try:
                    vectorstore._client.reset()  # 클라이언트 리셋
                except:
                    pass  # 리셋 실패해도 계속 진행
            del vectorstore  # 객체 삭제로 메모리 해제
        
        # 3단계: 세션 딕셔너리에서 제거
        del sessions[session_id]  # 메모리에서 세션 데이터 완전 제거
        
        # 4단계: 파일 시스템에서 비동기적으로 삭제
        session_dir = os.path.join(CHROMA_DB_DIR, session_id)  # 세션 디렉토리 경로
        if os.path.exists(session_dir):
            import asyncio  # 비동기 처리용
            import shutil   # 디렉토리 삭제용
            
            async def cleanup_directory():
                """
                세션 디렉토리를 비동기적으로 정리하는 내부 함수
                파일 잠금 문제 해결을 위한 재시도 로직 포함
                """
                max_retries = 3  # 최대 3번 재시도
                for attempt in range(max_retries):
                    try:
                        await asyncio.sleep(0.1)  # 짧은 대기 (파일 핸들 해제 대기)
                        shutil.rmtree(session_dir)  # 디렉토리 삭제
                        break  # 성공시 루프 종료
                    except PermissionError:
                        # 파일 권한 오류시 재시도
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.5)  # 재시도 전 대기
                        else:
                            print(f"⚠️ 세션 디렉토리 삭제 지연됨: {session_id}")
            
            # 백그라운드에서 삭제 작업 수행 (API 응답 지연 방지)
            asyncio.create_task(cleanup_directory())
        
        # 5단계: 성공 응답 반환
        return {"status": "success", "message": "Session deleted"}
        
    except Exception as e:
        # 예외 발생시 HTTP 오류로 변환
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

# ================================================================================================
# 텍스트 처리 유틸리티 함수들
# ================================================================================================

def remove_heading(text, heading):
    """
    텍스트에서 제목 부분을 제거하는 함수
    
    Args:
        text (str): 처리할 텍스트
        heading (str): 제거할 제목 (예: "개선 제안")
        
    Returns:
        str: 제목이 제거된 깔끔한 텍스트
    """
    # "개선 제안", "개선 제안:", "개선 제안 " 등 다양한 형태의 제목 제거
    for h in [heading, f"{heading}:", f"{heading} "]:
        if text.strip().startswith(h):
            # 제목과 콜론, 공백, 줄바꿈, 쌍따옴표 제거
            return text.strip()[len(h):].lstrip(": \n\"")
    return text.strip()  # 제목이 없는 경우 그대로 반환

def prettify_bullet(text, emoji="•"):
    """
    마크다운 스타일의 불릿 포인트를 이모지로 변환하는 함수
    
    Args:
        text (str): 변환할 텍스트
        emoji (str): 사용할 이모지 (기본값: "•")
        
    Returns:
        str: 이모지로 변환된 텍스트
    """
    lines = text.split('\n')  # 줄 단위로 분할
    pretty_lines = []
    for line in lines:
        if line.strip().startswith('- '):  # "- "로 시작하는 불릿 포인트 확인
            # "- " 제거하고 이모지로 교체
            pretty_lines.append(f"{emoji} {line.strip()[2:]}")
        else:
            pretty_lines.append(line.strip())  # 일반 텍스트는 그대로 유지
    return '\n'.join(pretty_lines)

def prettify_feedback(text):
    """
    피드백 텍스트의 가독성을 향상시키는 함수
    특정 키워드를 이모지로 변환하고 구조화
    
    Args:
        text (str): 원본 피드백 텍스트
        
    Returns:
        str: 이모지와 포맷팅이 적용된 텍스트
    """
    # 주요 키워드를 이모지로 변환하여 가독성 향상
    text = text.replace("장점:", "\n💡 ")        # 장점 → 전구 이모지
    text = text.replace("부족한 점:", "\n⚠️ ")   # 부족한 점 → 경고 이모지
    text = text.replace("개선필요:", "\n📝 ")     # 개선필요 → 메모 이모지
    text = text.replace("개선점:", "\n📝 ")       # 개선점 → 메모 이모지
    
    # 불릿 포인트를 화살표 이모지로 변환
    text = prettify_bullet(text, emoji="👉")
    return text.strip()

def prettify_evaluation(text):
    """
    평가 텍스트에 체크 이모지를 적용하는 함수
    
    Args:
        text (str): 원본 평가 텍스트
        
    Returns:
        str: 체크 이모지가 적용된 텍스트
    """
    return prettify_bullet(text, emoji="✅")  # 체크 이모지 사용

def prettify_suggestion(text):
    """
    제안 텍스트에서 불필요한 쌍따옴표를 제거하는 함수
    
    Args:
        text (str): 원본 제안 텍스트
        
    Returns:
        str: 쌍따옴표가 제거된 깔끔한 텍스트
    """
    return text.replace('"', '').strip()  # 쌍따옴표 제거 후 공백 정리

# ================================================================================================
# AI 체인 생성 함수 (벡터 검색 + Claude AI 조합)
# ================================================================================================

def create_chain(vectorstore):
    """
    벡터 검색과 Claude AI를 결합한 처리 체인을 생성하는 함수
    
    처리 과정:
    1. 벡터스토어에서 유사 문서 검색기 생성
    2. Claude AI 모델 초기화 (SSL 검증 비활성화)
    3. 프롬프트 템플릿과 모델을 연결하여 체인 구성
    
    Args:
        vectorstore: ChromaDB 벡터 데이터베이스 객체
        
    Returns:
        chain: LangChain 처리 체인 객체
    """
    try:
        # 1단계: 벡터 검색기 생성
        print("Creating retriever...")
        retriever = vectorstore.as_retriever(
            search_type="similarity",        # 코사인 유사도 기반 검색
            search_kwargs={"k": SEARCH_K}   # 상위 3개 유사 문서 반환
        )
        print("Retriever created successfully")
        
        # 2단계: Claude AI용 프롬프트 템플릿 정의
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

        # 3단계: 프롬프트 템플릿 객체 생성
        print("Creating prompt template...")
        prompt = ChatPromptTemplate.from_template(template)
        print("Prompt template created successfully")
        
        # 4단계: Anthropic API 키 확인
        api_key = os.getenv("ANTHROPIC_API_KEY")  # 환경변수에서 API 키 읽기
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not found")
        
        print("Creating ChatAnthropic model...")
        
        # 5단계: SSL 검증 비활성화 설정 (개발 환경용)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False  # 호스트명 검증 비활성화
        ssl_context.verify_mode = ssl.CERT_NONE  # 인증서 검증 비활성화
        
        # 전역 SSL 컨텍스트 설정
        ssl._create_default_https_context = lambda: ssl_context
        
        # 환경 변수를 통해 SSL 검증 비활성화
        os.environ['ANTHROPIC_VERIFY_SSL'] = 'false'
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_CERT_FILE'] = ''
        
        # 6단계: Claude AI 모델 객체 생성
        model = ChatAnthropic(
            model_name="claude-3-5-sonnet-20241022",  # 최신 Claude 3.5 Sonnet 모델
            temperature=0,                            # 일관된 응답을 위해 창의성 최소화
            api_key=SecretStr(api_key),              # API 키 (보안 처리)
            max_tokens_to_sample=2048,               # 최대 토큰 수 (긴 응답 허용)
            timeout=60,                              # API 타임아웃 60초
            stop=None                                # 특별한 중단 조건 없음
        )
        print("ChatAnthropic model created successfully")
        
        # 7단계: Anthropic 클라이언트의 SSL 검증 비활성화
        if hasattr(model, '_client') and hasattr(model._client, '_client'):
            # 기존 클라이언트 설정 복사
            old_client = model._client._client
            # SSL 검증 비활성화된 새 클라이언트 생성
            new_client = httpx.Client(
                verify=False,                         # SSL 검증 비활성화
                timeout=60.0,                        # 타임아웃 설정
                headers=old_client.headers,          # 기존 헤더 유지
                cookies=old_client.cookies,          # 기존 쿠키 유지
                auth=old_client.auth,                # 기존 인증 유지
                follow_redirects=old_client.follow_redirects  # 리다이렉트 설정 유지
            )
            # 내부 클라이언트 교체
            model._client._client = new_client
            print("Anthropic client SSL verification disabled successfully")
        
        # 8단계: LangChain 처리 체인 구성
        print("Creating chain...")
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}  # 검색 결과와 질문을 함께 전달
            | prompt  # 프롬프트 템플릿 적용
            | model   # Claude AI 모델로 처리
        )
        print("Chain created successfully")
        
        return chain  # 완성된 체인 반환
    except Exception as e:
        # 체인 생성 중 오류 발생시 상세 로그 출력
        import traceback
        print("=== CREATE CHAIN ERROR ===")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        print("==========================")
        raise HTTPException(status_code=500, detail=f"Chain 생성 중 오류 발생: {str(e)}")

# ================================================================================================
# 서버 종료 처리 함수
# ================================================================================================

def cleanup_all_sessions():
    """
    서버 종료 시 모든 활성 세션을 정리하는 함수
    
    기능:
    1. 모든 세션의 벡터스토어 객체 정리
    2. ChromaDB 클라이언트 연결 해제
    3. 메모리에서 세션 데이터 완전 제거
    
    주의: 서버 종료시 자동으로 호출됨 (atexit 등록)
    """
    print("🧹 서버 종료 중 - 모든 세션 정리...")
    
    # 모든 활성 세션을 순회하며 정리
    for session_id in list(sessions.keys()):  # 리스트 복사로 안전한 순회
        try:
            if 'vectorstore' in sessions[session_id]:
                vectorstore = sessions[session_id]['vectorstore']  # 벡터스토어 객체 획득
                # ChromaDB 클라이언트 연결 정리
                if hasattr(vectorstore, '_client') and vectorstore._client:
                    vectorstore._client.reset()  # 클라이언트 리셋
        except:
            pass  # 개별 세션 정리 실패해도 계속 진행
    
    sessions.clear()  # 세션 딕셔너리 완전 초기화

# ================================================================================================
# 서버 시작점
# ================================================================================================

# 서버 종료 시 정리 함수 등록 (프로그램 종료시 자동 호출)
atexit.register(cleanup_all_sessions)

# 메인 실행 블록 (스크립트가 직접 실행될 때만 서버 시작)
if __name__ == "__main__":
    import uvicorn  # ASGI 서버 (FastAPI 실행용)
    # uvicorn으로 FastAPI 앱 실행
    uvicorn.run(
        app,               # FastAPI 앱 객체
        host="0.0.0.0",    # 모든 IP에서 접근 허용
        port=8000          # 8000번 포트에서 서비스
    ) 