import os
import yaml
import hashlib
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import loading
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
import random


class PDFQuestionGenerator:
    def __init__(self, model_name="gpt-4o", temperature=0):
        """초기화: 환경 설정 및 모델 로드"""
        load_dotenv()
        os.environ["LANGCHAIN_PROJECT"] = "hackathon_test_model"
        os.environ["LANGSMITH_TRACING"] = "true"

        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        # 캐시 디렉토리 생성
        os.makedirs(".cache/files", exist_ok=True)
        os.makedirs(".cache/embeddings", exist_ok=True)

    def get_cache_path(self, file_path):
        """PDF 파일명을 기반으로 해시 생성 후 캐시 경로 반환"""
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        return f".cache/embeddings/{file_hash}"

    def load_pdf(self, file_path):
        """PDF 파일을 로드하고 모든 페이지의 텍스트를 하나의 리스트로 저장"""
        loader = PDFPlumberLoader(file_path)
        raw_docs = loader.load()

        print(
            f"총 {len(raw_docs)}개의 페이지를 로드했습니다."
        )  # ✅ 전체 페이지 개수 확인

        # ✅ `raw_docs`를 `Document` 객체 리스트로 변환
        docs = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in raw_docs
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200  # ✅ 페이지별로 최적화된 분할 크기
        )

        # ✅ `Document` 객체 리스트를 전달
        split_documents = text_splitter.split_documents(docs)

        print(
            f"총 {len(split_documents)}개의 조각으로 분할되었습니다."
        )  # ✅ 분할 개수 확인

        return split_documents

    def create_retriever(self, file_path, split_documents):
        """FAISS 벡터스토어를 캐싱하여 저장하고 필요 시 불러오는 retriever 생성"""
        cache_path = self.get_cache_path(file_path)

        if os.path.exists(cache_path):
            print("Loading cached FAISS embeddings...")
            vectorstore = FAISS.load_local(
                cache_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True
            )
        else:
            print("Creating new FAISS embeddings...")
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(
                documents=split_documents, embedding=embeddings
            )

            # FAISS 인덱스를 파일로 저장
            vectorstore.save_local(cache_path)

        return vectorstore.as_retriever()

    def get_prompt_template_path(self, difficulty, question_type):
        """주어진 난이도와 문제 유형에 따른 프롬프트 템플릿 경로 반환"""
        base_path = "pdf_problem_generator/prompts"
        mapping = {
            # ("객관식", "하"): "multiplechoice_low.yaml",
            ("객관식", "중"): "multiplechoice_medium.yaml",
            # ("객관식", "상"): "multiplechoice_high.yaml",
            # ("단답형", "하"): "shortanswer_low.yaml",
            ("단답형", "중"): "shortanswer_medium.yaml",
            # ("단답형", "상"): "shortanswer_high.yaml",
            # ("서술형", "하"): "essay_low.yaml",
            ("서술형", "중"): "essay_medium.yaml",
            # ("서술형", "상"): "essay_high.yaml",
        }
        return os.path.join(
            base_path,
            mapping.get((question_type, difficulty), "shortanswer_easy.yaml"),
        )

    def create_chain(self, num_questions, prompt_path):
        """문제 생성 LLM 체인 생성"""
        with open(prompt_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        raw_prompt = loading.load_prompt_from_config(config)

        prompt_runnable = RunnableLambda(lambda state: raw_prompt.format(**state))

        input_runnable = {
            "question": RunnableLambda(lambda state: state["question"]),
            "context": RunnableLambda(lambda state: state["context"]),
            "num_questions": RunnableLambda(lambda state: str(num_questions)),
        }

        chain = input_runnable | prompt_runnable | self.llm | StrOutputParser()
        return chain

    def generate_questions(
        self, pdf_path, num_questions, difficulty, question_type, user_message=""
    ):
        """
        PDF에서 문제를 생성하는 전체 프로세스 실행
        - 유사도 검색 기반 문서 검색 or 균등 샘플링 후 LLM 체인 실행
        - 참고한 페이지 번호도 함께 반환
        """
        # PDF 로드 및 텍스트 추출
        split_documents = self.load_pdf(pdf_path)

        # Retriever 생성 (FAISS 캐싱된 임베딩 활용)
        retriever = self.create_retriever(pdf_path, split_documents)

        # 프롬프트 경로 결정
        prompt_path = self.get_prompt_template_path(difficulty, question_type)
        print("Using prompt template:", prompt_path)

        # LLM 체인 생성
        chain = self.create_chain(num_questions, prompt_path)

        # 참고한 페이지 번호 저장할 set (중복 제거)
        referenced_pages = set()

        # 유저 입력이 있을 경우 retriever를 통해 검색
        if user_message.strip():
            retrieved_docs = retriever.invoke(user_message)
            context_text = "\n".join([doc.page_content for doc in retrieved_docs])

            # ✅ 참고한 문서들의 페이지 번호 저장
            for doc in retrieved_docs:
                if "page" in doc.metadata:
                    referenced_pages.add(doc.metadata["page"])

        else:
            # ✅ 랜덤 샘플링을 위해 추가
            # 유저 메시지가 없으면 랜덤 샘플링 수행
            total_docs = len(split_documents)
            sample_size = min(5, total_docs)  # 최대 5개까지 선택

            # ✅ 랜덤 샘플링: 0부터 total_docs-1까지 중에서 sample_size개의 인덱스 무작위 선택
            indices = random.sample(range(total_docs), sample_size)

            # ✅ 랜덤하게 선택된 인덱스를 기반으로 샘플링된 문서 선택
            sampled_docs = [split_documents[i] for i in indices]

            # ✅ 선택된 문서들을 연결하여 컨텍스트 생성
            context_text = "\n".join([doc.page_content for doc in sampled_docs])

            # ✅ 균등 샘플링된 문서들의 페이지 번호 저장
            for doc in sampled_docs:
                if "page" in doc.metadata:
                    referenced_pages.add(doc.metadata["page"])

        response = chain.invoke({"question": user_message, "context": context_text})

        # ✅ 참고한 페이지 번호 정렬하여 리스트로 변환
        referenced_pages = sorted(list(referenced_pages))

        # ✅ 문제와 참고한 페이지 번호 함께 반환
        return response, referenced_pages


if __name__ == "__main__":
    generator = PDFQuestionGenerator()

    # pdf_path = "/Users/jeongtaek/kakaotech/pdf_problem_generator/필기(요약).pdf"
    pdf_path = "https://amzn-s3-testprepai.s3.amazonaws.com/test.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAXDKDZV7A4DD4BYFT%2F20250226%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Date=20250226T092039Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=801a7208d4541e6d1fef8bb9d9b0a4eadeed81cd453dd67d6337dda264736c44"
    num_questions = 5
    difficulty = "하"
    question_type = "단답식"
    user_message = ""  # 기본값 (없으면 균등 샘플링)

    result, referenced_pages = generator.generate_questions(
        pdf_path, num_questions, difficulty, question_type, user_message
    )
    print("\n생성된 문제:\n")
    print(result)
    print(referenced_pages)
