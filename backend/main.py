from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Optional, List, Dict
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document
from langchain.text_splitter import MarkdownTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import ssl
import gc
import re
import uuid
from datetime import datetime, timedelta

# SSL 인증서 검증 비활성화
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()

# Configuration constants
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEARCH_K = 3
CHROMA_DB_DIR = "./chroma_db"
MODEL_CACHE_DIR = "./model_cache"
SESSION_TIMEOUT = timedelta(hours=1)  # 세션 타임아웃 설정

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영환경에서는 구체적인 origin을 지정해야 합니다
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

# Global variables for models and session management
tokenizer = None
model = None
sessions: Dict[str, dict] = {}  # session_id -> {vectorstore, created_at, area, academic_level}

def get_embeddings(text, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 생성 중 오류 발생: {str(e)}")

class CustomEmbeddingFunction(Embeddings):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def embed_documents(self, texts):
        results = []
        for text in texts:
            emb = get_embeddings(text, self.tokenizer, self.model)
            results.append(emb)
        return results

    def embed_query(self, text):
        return get_embeddings(text, self.tokenizer, self.model)

def download_and_cache_model():
    global tokenizer, model
    try:
        if not os.path.exists(MODEL_CACHE_DIR):
            os.makedirs(MODEL_CACHE_DIR)
        
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, cache_dir=MODEL_CACHE_DIR)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL, cache_dir=MODEL_CACHE_DIR)
        model = model.to('cpu')
        
        return tokenizer, model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 다운로드 중 오류 발생: {str(e)}")

def create_chain(vectorstore):
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": SEARCH_K}
        )
        
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

        prompt = ChatPromptTemplate.from_template(template)
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not found")
        
        model = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0, # ← 더 창의적인 결과를 원하면 0.3~0.7로 조정
            anthropic_api_key=api_key,
            max_tokens=2048
        )
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
        )
        
        return chain
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chain 생성 중 오류 발생: {str(e)}")

def cleanup_expired_sessions():
    """만료된 세션을 정리합니다."""
    current_time = datetime.now()
    expired_sessions = [
        session_id for session_id, session_data in sessions.items()
        if current_time - session_data['created_at'] > SESSION_TIMEOUT
    ]
    for session_id in expired_sessions:
        if os.path.exists(os.path.join(CHROMA_DB_DIR, session_id)):
            import shutil
            shutil.rmtree(os.path.join(CHROMA_DB_DIR, session_id))
        del sessions[session_id]

@app.post("/api/load-documents")
async def load_documents(request: LoadDocumentRequest):
    cleanup_expired_sessions()  # 만료된 세션 정리
    
    area_map = {
        "자율/자치활동 특기사항": "self_governance_guidelines",
        "진로활동 특기사항": "career_activity_guidelines"
    }
    
    if request.area not in area_map:
        raise HTTPException(status_code=400, detail=f"Invalid area selected: {request.area}")
    
    try:
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
        if tokenizer is None or model is None:
            try:
                download_and_cache_model()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")
        
        embedding_function = CustomEmbeddingFunction(tokenizer, model)
        
        try:
            # Create vectorstore for this session
            os.makedirs(session_db_dir, exist_ok=True)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding_function,
                persist_directory=session_db_dir
            )
            
            # Store session information
            sessions[session_id] = {
                'vectorstore': vectorstore,
                'created_at': datetime.now(),
                'area': request.area,
                'academic_level': request.academic_level
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating vectorstore: {str(e)}")
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "status": "success",
            "message": "Documents loaded successfully",
            "session_id": session_id
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}\n{error_details}")

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

        chain = create_chain(vectorstore)
        print("Claude API 호출 프롬프트 전체:")
        print(template.replace("{question}", request.statement))
        response = chain.invoke(request.statement)
        
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
        except Exception:
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

        return {
            "evaluation": eval_part,
            "feedback": feedback_part,
            "suggestion": suggestion_part,
            "suggestion_length": suggestion_length
        }

    except Exception as e:
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
    """특정 세션을 삭제합니다."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Chroma DB 디렉토리 삭제
        session_db_dir = os.path.join(CHROMA_DB_DIR, session_id)
        if os.path.exists(session_db_dir):
            import shutil
            shutil.rmtree(session_db_dir)
        
        # 세션 정보 삭제
        del sessions[session_id]
        
        return {"status": "success", "message": "Session deleted successfully"}
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
    text = text.replace("장점:", "\n💡 장점")
    text = text.replace("부족한 점:", "\n⚠️ 부족한 점")
    text = text.replace("개선필요:", "\n📝 개선 필요")
    text = text.replace("개선점:", "\n📝 개선점")
    text = prettify_bullet(text, emoji="👉")
    return text.strip()

def prettify_evaluation(text):
    return prettify_bullet(text, emoji="✅")

def prettify_suggestion(text):
    # 쌍따옴표 제거
    return text.replace('"', '').strip()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 