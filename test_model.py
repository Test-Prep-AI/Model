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
import time


class PDFQuestionGenerator:
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        """초기화: 환경 설정 및 모델 로드"""
        load_dotenv()
        # 랭스미스 트레이싱
        os.environ["LANGCHAIN_PROJECT"] = "hackathon_final_model"  # 프로젝트 명 설정
        os.environ["LANGSMITH_TRACING"] = "true"

        # 캐시 디렉토리 생성
        os.makedirs(".cache/embeddings", exist_ok=True)

        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

    def get_cache_path(self, file_path):
        """PDF 파일명을 기반으로 해시 생성 후 캐시 경로 반환"""
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        return f".cache/embeddings/{file_hash}"

    def load_pdf(self, file_path):
        """PDF 파일을 로드하고 모든 페이지의 텍스트를 하나의 리스트로 저장"""
        loader = PDFPlumberLoader(file_path)
        raw_docs = loader.load()

        # pdf 전체 페이지 개수 확인
        print(f"총 {len(raw_docs)}개의 페이지를 로드했습니다.")

        # `raw_docs`를 `Document` 객체 리스트로 변환
        docs = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in raw_docs
        ]

        # 청크 분할 크기 설정
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

        # `Document` 객체 리스트를 전달하여 청크 분할
        split_documents = text_splitter.split_documents(docs)

        # 분할된 청크(조각) 개수 확인
        print(f"총 {len(split_documents)}개의 조각으로 분할되었습니다.")

        return split_documents

    def create_retriever(self, file_path, split_documents):
        """FAISS 벡터스토어를 캐싱하여 저장하고 필요 시 불러오는 retriever 생성"""
        # pdf 캐시 경로 반환
        cache_path = self.get_cache_path(file_path)

        # 한 번 임베딩 된 pdf의 경우 캐시파일에서 불러오기, OpenAIEmbeddings() 형식
        if os.path.exists(cache_path):
            vectorstore = FAISS.load_local(
                cache_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True
            )
        else:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(
                documents=split_documents, embedding=embeddings
            )
            # FAISS 인덱스를 파일로 저장
            vectorstore.save_local(cache_path)

        return vectorstore.as_retriever()

    def get_prompt_template_path(self, difficulty, question_type):
        """주어진 난이도와 문제 유형에 따른 프롬프트 템플릿 경로 반환"""
        base_path = "prompts"
        mapping = {
            ("객관식", "하"): "multiplechoice_easy.yaml",
            ("객관식", "중"): "multiplechoice_medium.yaml",
            ("객관식", "상"): "multiplechoice_hard.yaml",
            ("단답형", "하"): "shortanswer_easy.yaml",
            ("단답형", "중"): "shortanswer_medium.yaml",
            ("단답형", "상"): "shortanswer_hard.yaml",
            ("서술형", "하"): "essay_easy.yaml",
            ("서술형", "중"): "essay_medium.yaml",
            ("서술형", "상"): "essay_hard.yaml",
        }
        return os.path.join(
            base_path,
            mapping.get((question_type, difficulty)),
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
        - 유저 메세지o -> 유사도 검색 기반 문서 검색 후 LLM 체인 실행
        - 유저 메세지x -> 랜덤 샘플링 문서 검색 후 LLM 체인 실행
        - 참고한 페이지 번호도 함께 반환
        """
        split_documents = self.load_pdf(pdf_path)
        retriever = self.create_retriever(pdf_path, split_documents)
        prompt_path = self.get_prompt_template_path(difficulty, question_type)
        chain = self.create_chain(num_questions, prompt_path)

        referenced_pages = set()

        if user_message.strip():
            retrieved_docs = retriever.invoke(user_message)
            context_text = "\n".join([doc.page_content for doc in retrieved_docs])
            for doc in retrieved_docs:
                if "page" in doc.metadata:
                    referenced_pages.add(doc.metadata["page"])
        else:
            total_docs = len(split_documents)
            sample_size = min(num_questions, total_docs)
            indices = random.sample(range(total_docs), sample_size)
            sampled_docs = [split_documents[i] for i in indices]
            context_text = "\n".join([doc.page_content for doc in sampled_docs])
            for doc in sampled_docs:
                if "page" in doc.metadata:
                    referenced_pages.add(doc.metadata["page"])

        response = chain.invoke({"question": user_message, "context": context_text})
        referenced_pages = sorted(list(referenced_pages))
        return response, referenced_pages

    def generate_overall_topic(self, pdf_path):
        """
        PDF의 전반적인 주제를 도출하는 기능:
        - PDF를 청크로 분할하고 각 청크별로 요약 생성
        - 생성된 요약들을 종합해 전반적인 주제를 한 문장으로 도출
        """
        split_documents = self.load_pdf(pdf_path)
        chunk_summaries = []

        # 랜덤 샘플링, 청크 5개 추출
        total_docs = len(split_documents)
        sample_size = min(8, total_docs)
        indices = random.sample(range(total_docs), sample_size)
        sampled_docs = [split_documents[i] for i in indices]

        # 각 청크에 대해 LLM으로 간단 요약 생성
        for i, doc in enumerate(sampled_docs):
            prompt = f"다음 PDF 내용에서 중요 키워드를 ','으로 구분해서 5개 추출해줘:\n\n{doc.page_content}\n\n"
            response = self.llm.invoke(prompt)
            # AIMessage 객체의 텍스트는 보통 .content 속성에 있음
            summary = (
                response.content if hasattr(response, "content") else str(response)
            )
            chunk_summaries.append(summary.strip())
            # print(f"청크 {i+1} 키워드:", summary.strip())

        # 모든 청크 요약을 종합하여 전반적인 주제 도출
        combined_summary = "\n".join(chunk_summaries)
        overall_prompt = (
            """
            다음은 PDF 각 부분의 키워드입니다. 이를 바탕으로 PDF가 어떤 큰 틀의 시험이나 과목에 관련된 자료인지 **명칭만 알려주세요.**
            (ex: 정보처리기사 관련 자료, 빅데이터분석기사 관련 자료, 선형대수학 행렬 관련 자료, 자바스크립트 관련 자료, git 명령어 관련 자료, NASA 관련 자료):\n\n
            """
            f"{combined_summary}\n\n 커리큘럼:"
        )
        overall_response = self.llm.invoke(overall_prompt)
        overall_topic = (
            overall_response.content
            if hasattr(overall_response, "content")
            else str(overall_response)
        )
        print(overall_topic)
        return overall_topic.strip()
        


if __name__ == "__main__":
    start = time.time()
    generator = PDFQuestionGenerator()

    pdf_path = "https://www.kimnbook.co.kr/uploads/board/reference/B_7902/b097a9f0c6683140ac29dd931d71274d.pdf"
    num_questions = 10
    difficulty = "중"
    question_type = "객관식"
    user_message = "자료구조"

    # 문제 생성
    result, referenced_pages = generator.generate_questions(
        pdf_path, num_questions, difficulty, question_type, user_message
    )
    print("\n생성된 문제:\n")
    print(result)
    print("참고한 페이지:", referenced_pages)

    # PDF 전반적인 주제 도출
    overall_topic = generator.generate_overall_topic(pdf_path)
    print("\nPDF의 전반적인 주제:")
    print(overall_topic)
    
    end = time.time()
    print(f"{end - start:.5f} sec")
