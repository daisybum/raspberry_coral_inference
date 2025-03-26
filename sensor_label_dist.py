import json
from collections import defaultdict, Counter

# 3개의 coco.json 파일 경로를 리스트로 지정
json_files = [
    '/media/shpark/new-volumn/merged_all/train_annotations.coco.json', 
    '/media/shpark/new-volumn/merged_all/val_annotations.coco.json', 
    '/media/shpark/new-volumn/merged_all/test_annotations.coco.json'
]

# 합칠 데이터를 위한 초기 구조 생성 (images, annotations, categories)
combined_data = {"images": [], "annotations": [], "categories": []}

# 각 파일에서 데이터를 읽어와 합치기 (categories는 동일하다고 가정하여 한 번만 저장)
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        combined_data["images"].extend(data.get("images", []))
        combined_data["annotations"].extend(data.get("annotations", []))
        if not combined_data["categories"]:
            combined_data["categories"] = data.get("categories", [])

# 센서 이름 추출 함수 (파일 이름에 "M"으로 시작하면 앞 3개 토큰 사용)
def get_sensor_name(filename):
    # 확장자 제거
    base = filename.rsplit('.', 1)[0]
    tokens = base.split('_')
    # 파일명이 "M"으로 시작하는 경우, 예: "MVW_B1_000003_20230504_163753.jpg"
    if base.startswith("MVW"):
        # 센서 이름은 앞 3개의 토큰 (예: MVW_B1_000003)
        sensor = '_'.join(tokens[:3])
    else:
        # 그 외의 경우 앞 4개 토큰을 센서 이름으로 사용
        sensor = '_'.join(tokens[:4])
    return sensor

# 이미지 ID와 센서 이름의 매핑 생성
image_sensor_mapping = {}
for image in combined_data["images"]:
    sensor = get_sensor_name(image["file_name"])
    image_sensor_mapping[image["id"]] = sensor

# 센서별로 annotation의 category_id 개수를 집계
sensor_annotation_counts = defaultdict(Counter)
for ann in combined_data["annotations"]:
    image_id = ann["image_id"]
    sensor = image_sensor_mapping.get(image_id)
    if sensor:
        sensor_annotation_counts[sensor][ann["category_id"]] += 1

# category id를 이름으로 매핑 (예: {1: "dry", 2: "humid", ...})
cat_id_to_name = {cat["id"]: cat["name"] for cat in combined_data["categories"]}

# 센서별 라벨 분포를 퍼센트로 계산하여 출력
for sensor, counter in sensor_annotation_counts.items():
    total = sum(counter.values())
    print(f"센서: {sensor}")
    for cat_id, count in counter.items():
        percentage = (count / total) * 100
        cat_name = cat_id_to_name.get(cat_id, str(cat_id))
        print(f"  {cat_name}: {percentage:.2f}%")
    print("-" * 40)
