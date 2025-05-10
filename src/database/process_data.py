import json
import sqlite3
import random

conn = sqlite3.connect('data/midjourney_styles_demo.db')

# Create a table
conn.execute('''
    CREATE TABLE IF NOT EXISTS styles (
        id TEXT PRIMARY KEY,
        name_zh TEXT,
        name_en TEXT,
        categories_zh TEXT,
        categories_en TEXT,
        features_zh TEXT,
        features_en TEXT,
        slug TEXT,
        slug_new TEXT,
        img_url TEXT,
        createdAt TEXT,
        promptBasic TEXT,
        type_zh TEXT,
        type_en TEXT,
        desc_zh TEXT,
        desc_en TEXT,
        ai_desc_zh TEXT,
        ai_style_zh TEXT,
        ai_features_zh TEXT,
        ai_color_zh TEXT,
        ai_desc_en TEXT,
        ai_style_en TEXT,
        ai_features_en TEXT,
        ai_color_en TEXT
    )
''')


# Insert data
def insert_data(data):
    conn.execute("""
    INSERT INTO styles (id, name_zh, name_en, categories_zh, categories_en, features_zh, features_en, slug, slug_new, img_url, createdAt, promptBasic, type_zh, type_en, desc_zh, desc_en, ai_desc_zh, ai_style_zh, ai_features_zh, ai_color_zh, ai_desc_en, ai_style_en, ai_features_en, ai_color_en)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(data['id']) + str(data["slug_new"]), data['name_zh'], data['name_en'], data['categories_zh'],
        data['categories_en'],
        data['features_zh'], data['features_en'], data['slug'], data['slug_new'], data['img_url'],
        data['createdAt'], data['promptBasic'], data['type_zh'], data['type_en'], data['desc_zh'],
        data['desc_en'], data['ai_desc_zh'], data['ai_style_zh'], data['ai_features_zh'],
        data['ai_color_zh'], data['ai_desc_en'], data['ai_style_en'], data['ai_features_en'],
        data['ai_color_en']
    ))
    conn.commit()


# 批量插入数据
def insert_data_batch(data_list):
    for data in data_list:
        insert_data(data)
    conn.commit()
    print("Data inserted successfully.")


# Load data from JSONL file and insert into database
def get_batch_data(file_path, batch_size):
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        # 随机打乱数据
        random.shuffle(lines)

        for i in range(0, len(lines), batch_size):
            batch_data = []
            for line in lines[i:i + batch_size]:
                data = eval(line)
                batch_data.append(data)
            yield batch_data


# sqlite3 数据库查询
def query_data():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM styles")
    print("Querying data from database...")
    rows = cursor.fetchall()
    print("Data count:", len(rows))
    for row in rows:
        print(row)
    cursor.close()


# 按照id查询
def query_by_id(id):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM styles WHERE id=?", (id,))
    row = cursor.fetchone()
    if row:
        print(row)
    else:
        print("No data found for id:", id)
    cursor.close()



# 批量插入数据
batch_size = 1000
file_path = 'data/midjoury_styles_lib_final_zh_en_demo.jsonl'
for batch_data in get_batch_data(file_path, batch_size):
    insert_data_batch(batch_data)

# 查询数据
# query_data()

# 查询单条数据
# query_by_id('1')
