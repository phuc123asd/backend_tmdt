import os
import json
import openai
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from astrapy import DataAPIClient
from sentence_transformers import SentenceTransformer

# ==============================================================================
# PHẦN 1: CẤU HÌNH
# ==============================================================================
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Thông tin kết nối Astra DB
ASTRA_TOKEN = os.getenv('ASTRA_TOKEN')
ASTRA_ENDPOINT = os.getenv('ASTRA_ENDPOINT')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

# Tên các model đang sử dụng
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')
# ==============================================================================
# PHẦN 2: KHỞI TẠO TOÀN CỤC (CHẠY MỘT LẦN KHI START SERVER)
# ==============================================================================

print("Đang khởi tạo các dịch vụ cho Chatbot...")
try:
    # Khởi tạo model Embedding
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Khởi tạo client của OpenAI
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    print("Đã khởi tạo Model Embedding và OpenAI Client thành công!")
except Exception as e:
    print(f"Lỗi nghiêm trọng khi khởi tạo: {e}")
    print("Chatbot sẽ không hoạt động. Vui lòng kiểm tra lại API keys và kết nối mạng.")
    embedding_model = None
    openai_client = None


# ==============================================================================
# PHẦN 3: LOGIC RAG CORE
# ==============================================================================

def get_rag_answer(question: str) -> str:
    """
    Hàm thực hiện toàn bộ luồng RAG:
    1. Truy vấn Astra DB để lấy ngữ cảnh.
    2. Tạo prompt cho OpenAI.
    3. Gọi OpenAI và trả về câu trả lời.
    """
    if not embedding_model or not openai_client:
        return "Dịch vụ chatbot chưa được khởi tạo đúng cách. Vui lòng kiểm tra lại log của server."

    try:
        # --- 1. RETRIEVAL (Truy vấn dữ liệu từ Astra DB) ---
        client = DataAPIClient(ASTRA_TOKEN)
        db = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)
        collection = db.get_collection(COLLECTION_NAME)

        # Tạo embedding cho câu hỏi của người dùng
        question_embedding = embedding_model.encode(question).tolist()

        # Tìm kiếm các vector tương tự nhất. Lấy 3 kết quả để có ngữ cảnh phong phú.
        results = collection.find(
            sort={"$vector": question_embedding},
            limit=3,
            include_similarity=True
        )

        if not results:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu của quán."

        # --- 2. AUGMENTED (Tạo prompt có ngữ cảnh) ---
        context_parts = []
        for res in results:
            # Chỉ lấy phần 'data' (thông tin gốc) để làm ngữ cảnh
            context_parts.append(json.dumps(res.get('data', {}), ensure_ascii=False))
        
        context_string = "\n---\n".join(context_parts)

        # Tạo prompt chi tiết cho LLM
        system_prompt = (
            "Bạn là một nhân viên tư vấn thân thiện và chuyên nghiệp cho một hệ thống cửa hàng bảo dưỡng xe máy. "
            "Nhiệm vụ của bạn là trả lời câu hỏi của khách hàng DỰA TRÊN KIẾN THỨC được cung cấp trong phần 'Ngữ cảnh' bên dưới. "
            "Hãy dùng ngôn ngữ tự nhiên, lịch sự. "
            "Nếu ngữ cảnh không chứa câu trả lời, hãy lịch sự nói rằng bạn không có thông tin đó. "
            "Tuyệt đối không bịa ra thông tin không có trong ngữ cảnh."
        )
        
        user_prompt = f"""
        --- Ngữ cảnh từ cơ sở dữ liệu ---
        {context_string}
        --- Hết ngữ cảnh ---

        Câu hỏi của khách hàng: {question}
        """
        
        # --- 3. GENERATION (Gọi LLM để tạo câu trả lời) ---
        response = openai_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2 # Giữ câu trả lời tập trung vào sự thật
        )
        
        final_answer = response.choices[0].message.content.strip()
        return final_answer

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý RAG: {e}")
        return "Xin lỗi, đã xảy ra lỗi hệ thống khi tôi đang tìm kiếm câu trả lời. Vui lòng thử lại sau."


# ==============================================================================
# PHẦN 4: DJANGO VIEW (Endpoint API)
# ==============================================================================

@api_view(['POST'])
def chat(request):
    """
    API endpoint cho chatbot.
    Nhận một câu hỏi trong body của POST request và trả về câu trả lời từ RAG.
    """
    # Lấy câu hỏi từ body của request
    question = request.data.get('question', '').strip()

    if not question:
        return Response(
            {"error": "Vui lòng cung cấp câu hỏi trong trường 'question'."},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Gọi hàm RAG để lấy câu trả lời
    answer = get_rag_answer(question)

    # Trả về kết quả
    return Response({"answer": answer}, status=status.HTTP_200_OK)
