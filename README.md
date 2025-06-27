# LifeRecordReview

생기부 특기사항 작성 검토 프로그램

## 프로젝트 소개
이 프로그램은 고등학교 생기부 특기사항 작성을 도와주는 AI 기반 검토 도구입니다. 
작성한 특기사항을 입력하면 AI가 문장을 분석하고 개선점을 제안합니다.

## 기술 스택
- python : v3.12
- Frontend: React, TypeScript
- Backend: FastAPI, Python
- AI: Anthropic Claude, LangChain
- Vector DB: ChromaDB

## 필수 프로그램
VS Code / Cursor AI (IDE)
Git
node.js

## 실행 방법

### 0. github 복제
git clone https://github.com/lifeprofessor/LifeRecordReviewSystem.git

### 1. 백엔드 실행
```bash
# backend 디렉토리로 이동
cd backend

# 가상환경 생성
#python -m venv venv

# 가상환경 활성화
# Windows의 경우:
#venv\Scripts\activate
# Linux/Mac의 경우:
#source venv/bin/activate

# 가상환경이 활성화되면 프롬프트 앞에 (venv)가 표시됩니다
# 이제 필요한 패키지들을 설치합니다
pip install -r requirements.txt

# 서버 실행 (상황별 선택)
# 1. 로컬 개발용 (같은 컴퓨터에서만 접근)
uvicorn main:app --host 127.0.0.1 --port 8000
# 또는
uvicorn main:app --port 8000

# 2. 같은 네트워크 내 다른 기기에서 접근 가능
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. 다른 포트 사용이 필요한 경우
uvicorn main:app --host 0.0.0.0 --port 3000  # 포트 번호 변경

# 4. ONNX Runtime + CUDA 방식으로 전환
## PyTorch 완전 제거
pip uninstall torch torchvision torchaudio -y

## ONNX Runtime + CUDA 설치
pip install onnxruntime-gpu
pip install transformers[onnx]
pip install optimum[onnxruntime]
pip install numpy

```

### 2. 프론트엔드 실행
```bash
# 이제 필요한 패키지들을 설치합니다
pip install -r requirements.txt

# frontend 디렉토리로 이동
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm start
```

## 환경 변수 설정
백엔드에서 필요한 환경 변수:
- `ANTHROPIC_API_KEY`: Anthropic API 키

## API 접속 주소 설정
프론트엔드에서 백엔드 API를 호출할 때 사용할 주소:

1. **로컬 개발 시**:
   - `http://localhost:8000/api/...`
   - `http://127.0.0.1:8000/api/...`

2. **같은 네트워크 내 다른 기기에서 접근 시**:
   - `http://[서버IP]:8000/api/...`
   - 예: `http://192.168.0.100:8000/api/...`

3. **외부에서 접근 시**:
   - `http://[공인IP]:8000/api/...`
   - DDNS 사용 시: `http://[도메인]:8000/api/...`

## 주의사항
1. **API 키 보안**
   - Anthropic API 키는 반드시 환경 변수로 관리
   - API 키를 코드에 직접 입력하지 않기

2. **데이터 관리**
   - ChromaDB 데이터는 정기적으로 백업
   - model_cache 디렉토리는 .gitignore에 포함되어 있음

3. **메모리 사용**
   - 임베딩 모델 로딩 시 메모리 사용량이 높을 수 있음
   - 충분한 RAM이 필요 (최소 8GB 권장)

4. **인터넷 연결**
   - AI 모델 사용을 위해 안정적인 인터넷 연결 필요
   - API 호출 실패 시 적절한 에러 처리 필요

5. **서버 접근 보안**
   - `--host 0.0.0.0` 사용 시 외부 접근이 가능하므로 주의
   - 필요한 경우 방화벽 설정
   - 프로덕션 환경에서는 SSL 인증서 사용 권장

## 배포 시 고려사항
1. **로컬 개발 환경**
   - localhost에서 프론트엔드와 백엔드가 통신
   - CORS 설정이 되어 있어 로컬 개발 가능

2. **웹 배포 시**
   - 포트포워딩 설정 필요
   - DDNS 서비스 사용 권장 (IP 변경 대응)
   - 보안을 위한 SSL 인증서 설정 권장

## 문제 해결
1. **모델 로딩 실패**
   - 인터넷 연결 확인
   - model_cache 디렉토리 권한 확인
   - 충분한 디스크 공간 확인

2. **API 연결 실패**
   - 환경 변수 설정 확인
   - API 키 유효성 확인
   - 네트워크 연결 상태 확인

3. **서버 접속 문제**
   - 포트가 사용 중인 경우: 다른 포트 사용
   - 방화벽 설정 확인
   - 네트워크 연결 상태 확인

4. **패키지 설치 문제**
   - 가상환경이 활성화되어 있는지 확인 ((venv) 프롬프트 확인)
   - pip가 최신 버전인지 확인: `pip install --upgrade pip`
   - 설치 실패 시: `pip install -r requirements.txt --no-cache-dir`

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

## 서비스 순서
[사용자 입력]
  └─> React 앱 (App.tsx)
       └─ POST /api/load-documents ─> load_documents()
             └─ 마크다운 로딩 + 임베딩 + Chroma 저장
       └─ POST /api/review ─> review_statement()
             └─ 벡터 검색 + Claude 프롬프트 평가
                   └─ 평가 + 예시문장 반환


### 코드의 실행 결과 예측 및 팁
예상 흐름 예시
1. 사용자가 자율/자치활동 특기사항, 상위권, 문장을 입력
2. 서버에서 관련 마크다운 불러와 KoSimCSE 임베딩 후 벡터 저장소 생성
3. 해당 저장소를 통해 의미 유사한 기존 문장 3개 검색
4. Claude에 맞춤 프롬프트와 함께 전달하여 평가 결과 + 추천 예시문장 생성
5. 결과는 클라이언트 화면에 표시

---


# 🏫 내부망 React 프론트엔드 배포 가이드

이 프로젝트는 학교/기관 내부망에서 React 기반 프론트엔드를 실행하고, FastAPI 백엔드와 연동하는 방법을 안내합니다. 외부 인터넷 없이도 브라우저에서 접근 가능하며, 백엔드와 안전하게 통신할 수 있도록 구성됩니다.

---

## 📁 프로젝트 구성 요약

```
project-root/
├── frontend/        # React 앱
├── backend/         # FastAPI 백엔드
```

---

## 🛠️ 환경 요구사항

- Node.js 및 npm 설치
- Python + FastAPI (백엔드용)
- 동일 네트워크 내 클라이언트 (교사망/학생망 등)

---

## 🚀 1. 프론트엔드 빌드 및 실행

### 1.1 의존성 설치

```bash
cd frontend
npm install
```

### 1.2 앱 빌드

```bash
npm run build
```

- `/frontend/build` 디렉토리가 생성됩니다.
- 이 디렉토리 안에 HTML, JS, CSS 등 정적 파일이 저장됩니다.

---

## 🌐 2. 정적 서버로 앱 실행 (`serve` 사용)

### 2.1 `serve` 설치

```bash
npm install -g serve
```

### 2.2 앱 실행

```bash
serve -s build -l 3000
```

- `-s`: SPA 모드 (React Router용)
- `-l 3000`: 3000번 포트로 실행

실행 후 아래와 같은 메시지를 확인할 수 있습니다:

```
Serving!
- Local:            http://localhost:3000
- On Your Network:  http://192.168.0.100:3000
```

---

## 📲 3. 다른 기기에서 접속하기

같은 네트워크에 있는 다른 PC, 태블릿, 스마트폰 등에서 다음과 같이 접속할 수 있습니다:

```
http://192.168.0.100:3000
```

- 위 IP 주소는 serve 실행 시 표시되는 실제 PC의 내부 IP로 변경해 주세요.
- 브라우저는 `http://` 접속이어야 하며, HTTPS가 아닙니다.

---

## 🔁 4. 백엔드(FastAPI)와 연동 설정

### 4.1 `.env` 파일에 API 주소 지정

`frontend/.env` 파일 생성 후:

```env
REACT_APP_API_URL=http://192.168.0.100:8000
```

> 백엔드는 `uvicorn` 등으로 HTTP 포트 8000에서 실행되고 있어야 합니다.

### 4.2 백엔드 실행 예시

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🧪 5. 실행 확인

- 브라우저에서 `http://<IP>:3000` 접속 → 앱이 잘 열리는지 확인
- 콘솔에 fetch 관련 에러가 없으면 백엔드 API 연결도 정상
- 백엔드 API도 브라우저에서 직접 접속 가능:  
  `http://<IP>:8000/docs` (FastAPI 문서)

---

## ⚙️ 6. 자동 실행 스크립트 (선택)

### 📁 Windows용 (`start_frontend.bat`)
```bat
@echo off
cd /d %~dp0
cd frontend
npm run build
serve -s build -l 3000
```

### 🐧 macOS/Linux용 (`start_frontend.sh`)
```bash
#!/bin/bash
cd "$(dirname "$0")/frontend"
npm run build
serve -s build -l 3000
```

> 실행 후 브라우저에서 http://<IP>:3000 접속

---

## 🔐 참고: HTTPS는 꼭 필요한가요?

- ❌ 내부망에서만 사용한다면 HTTPS는 **필수 아님**
- ✅ 단, 프론트엔드를 Netlify 등에 배포한 경우는 HTTPS → 반드시 백엔드도 HTTPS 필요

---

## 📚 관련 도구

| 도구       | 설명                                     |
|------------|------------------------------------------|
| React      | 프론트엔드 SPA (Single Page Application) |
| FastAPI    | Python 기반 백엔드 프레임워크            |
| serve      | 정적 웹서버. React 빌드 결과를 배포함     |
| uvicorn    | FastAPI 실행용 ASGI 서버                 |

---

## 📌 요약

| 항목            | 명령어 또는 경로                       |
|-----------------|----------------------------------------|
| 앱 빌드         | `npm run build`                        |
| 앱 실행         | `serve -s build -l 3000`               |
| 백엔드 실행     | `uvicorn main:app --host 0.0.0.0 --port 8000` |
| 내부망 접속 주소 | `http://<내부 IP>:3000`               |

---

내부망 환경에 맞게 설정 파일을 조정하고, 보안이 중요한 경우 인증 등 추가 기능을 구성하세요. 필요 시 자동화 및 배포 스크립트를 함께 제공합니다.
