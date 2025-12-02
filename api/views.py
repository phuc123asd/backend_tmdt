import os
import json
from openai import OpenAI
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from astrapy import DataAPIClient
import dotenv

dotenv.load_dotenv()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_TOKEN = os.getenv("ASTRA_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_ENDPOINT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# --------------------------------------------------
# INIT CLIENTS
# --------------------------------------------------
client_ai = OpenAI(api_key=OPENAI_API_KEY)
client_astra = DataAPIClient(ASTRA_TOKEN)
astra_db = client_astra.get_database_by_api_endpoint(ASTRA_ENDPOINT)
astra_collection = astra_db.get_collection(COLLECTION_NAME)


# --------------------------------------------------
# RAG CORE
# --------------------------------------------------
def get_rag_answer(question: str):

    print("\n================ RAG PIPELINE START ================")
    print(" User:", question)

    try:
        # ----------------------------------------------
        # STEP 1 — Embedding
        # ----------------------------------------------
        emb = client_ai.embeddings.create(
            model=EMBED_MODEL,
            input=question
        ).data[0].embedding

        print("Embedding created (1536 dims)")

        # ----------------------------------------------
        # STEP 2 — VECTOR SEARCH (SỬ DỤNG find() VỚI $vector SORT – CÁCH ĐÚNG CỦA ASTRAPY)
        # ----------------------------------------------
        print("Searching documents by vector similarity...")

        # Sử dụng find() với sort={"$vector": emb} – đây là API chuẩn cho vector search
        cursor = astra_collection.find(
            sort={"$vector": emb},  # Vector similarity sort
            limit=5,
            include_similarity=True  # Lấy điểm tương đồng (nếu hỗ trợ)
        )

        results = list(cursor)
        print(f"📌 Found {len(results)} matching docs")

        if len(results) == 0:
            print("⚠ No RAG match found")
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

        # ----------------------------------------------
        # STEP 3 — Build context
        # ----------------------------------------------
        context_parts = []

        print("\nExtracting context...")
        for doc in results:
            # Similarity ở '$similarity' với find() vector
            similarity = doc.get('$similarity', 0)
            print(f"✔ Context item (similarity: {similarity}): {doc.get('data', {})}")
            d = doc.get("data", {})
            context_parts.append(json.dumps(d, ensure_ascii=False))

        context = "\n---\n".join(context_parts)

        # ----------------------------------------------
        # STEP 4 — Build prompt
        # ----------------------------------------------
        system_prompt = (
            "Bạn là nhân viên tư vấn của hệ thống sửa xe máy. "
            "Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp. "
            "Nếu không có thông tin, hãy nói lịch sự rằng bạn không biết."
        )

        user_prompt = f"""
        --- NGỮ CẢNH ---
        {context}
        --- END ---

        Câu hỏi khách hàng: {question}
        """

        print("\nSending prompt to OpenAI...")

        # ----------------------------------------------
        # STEP 5 — Call ChatGPT
        # ----------------------------------------------
        resp = client_ai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        answer = resp.choices[0].message.content.strip()

        print("LLM ANSWER:", answer)
        print("================ RAG PIPELINE END ================\n")
        return answer

    except Exception as e:
        print("RAG ERROR:", e)
        import traceback
        traceback.print_exc()  # Để debug chi tiết
        return "Đã xảy ra lỗi khi xử lý yêu cầu. Vui lòng thử lại sau."


# --------------------------------------------------
# API ENDPOINT
# --------------------------------------------------
@api_view(["POST"])
def chat(request):
    question = request.data.get("question", "").strip()

    if not question:
        return Response({"error": "Missing field 'question'."}, status=400)

    answer = get_rag_answer(question)
    return Response({"answer": answer})