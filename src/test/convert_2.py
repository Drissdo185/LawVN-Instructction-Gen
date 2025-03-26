import json

def transform_format(item):
    """
    Chuyển đổi một item từ định dạng ban đầu sang định dạng mới
    
    Input: dictionary chứa category, mức_phạt và nội_dung
    Output: chuỗi có dạng "mức_phạt + " " + category + ": " + nội_dung"
    """
    category = item.get("category", "")
    muc_phat = item.get("mức_phạt", "")
    noi_dung_list = item.get("nội_dung", [])
    
    result = []
    for noi_dung in noi_dung_list:
        transformed_text = f"{muc_phat} {category}: {noi_dung}"
        result.append(transformed_text)
    
    return result

def main():
    # Đọc file JSON input
    with open('/home/drissdo/Desktop/LawVN-Instructction-Gen/src/data/rag_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Kiểm tra nếu data là một list
    if isinstance(data, list):
        all_transformed_data = []
        for item in data:
            # Chuyển đổi từng item trong list
            item_transformed = transform_format(item)
            all_transformed_data.extend(item_transformed)
    else:
        # Nếu data là một dictionary đơn lẻ
        all_transformed_data = transform_format(data)
    
    # Ghi ra file output
    with open('rag_data_transformed.json', 'w', encoding='utf-8') as f:
        json.dump(all_transformed_data, f, ensure_ascii=False, indent=2)
    
    print("Chuyển đổi hoàn tất. Dữ liệu đã được lưu vào file 'rag_data_transformed.json'")

if __name__ == "__main__":
    main()