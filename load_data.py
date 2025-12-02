import os
import json
from openai import OpenAI
from astrapy import DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.info import CollectionDefinition, CollectionVectorOptions
import dotenv

dotenv.load_dotenv()

# ----------------------------------------
# CONFIG
# ----------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_TOKEN = os.getenv("ASTRA_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_ENDPOINT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
FOLDER_PATH = r"C:\Users\vphuc\Downloads\DB"

# OpenAI client
client_ai = OpenAI(api_key=OPENAI_API_KEY)

# Astra client
client_astra = DataAPIClient(ASTRA_TOKEN)
db = client_astra.get_database_by_api_endpoint(ASTRA_ENDPOINT)


# ----------------------------------------
# CLEAN MONGO FIELDS
# ----------------------------------------
def clean_mongo_fields(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            if key.startswith("$"):  # remove $oid, $date...
                continue

            if key == "image" and isinstance(value, str) and len(value) > 8000:
                continue  # skip large images

            new_obj[key] = clean_mongo_fields(value)
        return new_obj

    if isinstance(obj, list):
        return [clean_mongo_fields(v) for v in obj]

    return obj


# ----------------------------------------
# CREATE COLLECTION (VECTOR ENABLED) – TỰ ĐỘNG XÓA NẾU THIẾU VECTOR
# ----------------------------------------
print("Checking collection...")
existing_collections = db.list_collection_names()

collection = None
if COLLECTION_NAME in existing_collections:
    print(f"✔ Collection '{COLLECTION_NAME}' exists. Checking vector support...")
    temp_coll = db.get_collection(COLLECTION_NAME)
    opts = temp_coll.options()  # Lấy schema
    if not opts.vector:
        print("Collection thiếu vector support! Đang xóa để tạo lại...")
        db.delete_collection(COLLECTION_NAME)
        existing_collections = db.list_collection_names()  # Refresh list
    else:
        print(f"Vector confirmed: dimension={opts.vector.dimension}, metric={opts.vector.metric.value}")
        collection = temp_coll

if COLLECTION_NAME not in existing_collections:
    print(f"Creating vector-enabled collection: {COLLECTION_NAME} ...")

    # Definition chuẩn cho v2.1.0: Sử dụng classes, không dict
    vector_opts = CollectionVectorOptions(
        dimension=1536,  # Match text-embedding-3-small
        metric=VectorMetric.COSINE
    )
    definition = CollectionDefinition(vector=vector_opts)

    collection = db.create_collection(
        COLLECTION_NAME,
        definition=definition
    )
    print("Vector-enabled collection created!")
else:
    collection = db.get_collection(COLLECTION_NAME)


# ----------------------------------------
# PROCESS AND UPLOAD JSON FILES (Giữ nguyên)
# ----------------------------------------
total_uploaded = 0
for filename in os.listdir(FOLDER_PATH):
    if not filename.endswith(".json"):
        continue

    type_name = filename.replace(".json", "")
    file_path = os.path.join(FOLDER_PATH, filename)

    print(f"\nFILE: {filename}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"Failed to read file {filename}: {e}")
        continue

    if not isinstance(raw_data, list):
        raw_data = [raw_data]

    print(f"Uploading {len(raw_data)} objects...")

    for index, item in enumerate(raw_data):
        cleaned = clean_mongo_fields(item)

        text = json.dumps(cleaned, ensure_ascii=False)

        # GET OPENAI EMBEDDING
        try:
            emb = client_ai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            continue

        # BUILD DOCUMENT
        doc = {
            "type": type_name,
            "data": cleaned,
            "$vector": emb  # VECTOR FIELD
        }

        # INSERT INTO ASTRA
        try:
            inserted_id = collection.insert_one(doc)
            print(f"   ✔ Inserted {index+1}/{len(raw_data)} → ID: {inserted_id}")
            total_uploaded += 1
        except Exception as e:
            print(f"Insert failed: {e}")

print(f"\nDONE: {total_uploaded} documents uploaded successfully!")
print("Bây giờ chạy test RAG API để kiểm tra vector search.")