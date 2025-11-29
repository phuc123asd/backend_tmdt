import os
import json
from astrapy import DataAPIClient
from sentence_transformers import SentenceTransformer

def clean_mongo_fields(obj):
    """Xóa các key kiểu $oid, $date để Astra chấp nhận và xử lý các trường quá lớn."""
    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            if key == "$oid":
                return value
            if key.startswith("$"):
                continue
            
            # Xử lý trường 'image' để tránh lỗi giới hạn kích thước
            if key == "image" and isinstance(value, str) and len(value.encode('utf-8')) > 8000:
                # Lựa chọn 1: Cắt bớt trường image
                # new_obj[key] = value[:8000] + "... [truncated]"
                
                # Lựa chọn 2: Bỏ qua trường image hoàn toàn (đang dùng)
                continue
                
            new_obj[key] = clean_mongo_fields(value)
        return new_obj

    if isinstance(obj, list):
        return [clean_mongo_fields(v) for v in obj]

    return obj


# ---------------------------
# 1. Config
# ---------------------------
ASTRA_TOKEN = os.getenv('ASTRA_TOKEN')
ASTRA_ENDPOINT = os.getenv('ASTRA_ENDPOINT')
FOLDER_PATH = r"C:\Users\vphuc\Downloads\DB"
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

# Model 384 chiều -> HOÀN TOÀN TƯƠNG THÍCH VỚI ASTRA
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------
# 2. Kết nối Astra DB
# ---------------------------
client = DataAPIClient(ASTRA_TOKEN)
db = client.get_database_by_api_endpoint(ASTRA_ENDPOINT)


# ---------------------------
# 3. Tạo collection với dimension 384
# ---------------------------
existing_collections = db.list_collection_names()

if COLLECTION_NAME not in existing_collections:
    print(f"🔧 Collection '{COLLECTION_NAME}' chưa tồn tại -> tạo mới...")
    db.create_collection(
        COLLECTION_NAME,
        definition={
            "vector": {
                "dimension": 384,
                "metric": "cosine"
            }
        }
    )
    print(f"✅ Collection '{COLLECTION_NAME}' đã được tạo!")
else:
    print(f"✔ Collection '{COLLECTION_NAME}' đã tồn tại.")

collection = db.get_collection(COLLECTION_NAME)


# ---------------------------
# 4. Upload từng file JSON, xử lý TỪNG ĐỐI TƯỢNG
# ---------------------------
for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".json"):
        file_path = os.path.join(FOLDER_PATH, filename)
        type_name = filename.replace(".json", "")

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ Lỗi đọc file JSON: {filename}. Bỏ qua.")
                continue

        # Đảm bảo raw_data là một danh sách các đối tượng
        if not isinstance(raw_data, list):
            # Nếu file chỉ chứa một đối tượng duy nhất, đặt nó vào một danh sách
            raw_data = [raw_data]
        
        if not raw_data:
            print(f"⚠️ File {filename} rỗng hoặc không có đối tượng nào. Bỏ qua.")
            continue

        print(f"\n📂 Đang xử lý file: {filename} với {len(raw_data)} đối tượng...")

        for i, item in enumerate(raw_data):
            # 1. Làm sạch từng đối tượng
            cleaned_item = clean_mongo_fields(item)

            # 2. Tạo văn bản để embedding từ TỪNG đối tượng
            text = json.dumps(cleaned_item, ensure_ascii=False)

            # 3. Tạo embedding 384-D cho TỪNG đối tượng
            embedding = model.encode(text).tolist()

            # 4. Tạo tài liệu để chèn vào Astra
            doc = {
                "type": type_name,
                "data": cleaned_item,
                "embedding": embedding
            }

            # 5. Chèn tài liệu
            try:
                inserted_id = collection.insert_one(doc)
                print(f"  ✅ Đã upload đối tượng {i+1}/{len(raw_data)} -> id = {inserted_id}")
            except Exception as e:
                print(f"  ❌ Lỗi khi upload đối tượng {i+1} từ {filename}: {str(e)}")
                # Tiếp tục với đối tượng tiếp theo trong file
                continue

print("\n🎉 HOÀN TẤT — Đã upload thành công tất cả các đối tượng!")