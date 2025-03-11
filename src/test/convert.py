import re
import json

def parse_traffic_violations(text):
    # Define categories and violation types
    category = "ô_tô"
    violation_types = [
        "trẻ_em", "tốc_độ", "nồng_độ_cồn", "ma_túy", "đỗ_dừng_xe", "giấy_tờ", 
        "đường_cao_tốc", "biển_báo", "làn_đường", "chở_người", "chở_hàng", 
        "điện_thoại", "lái_xe_nguy_hiểm", "xe_máy_đặc_thù", "kỹ_thuật_xe", 
        "môi_trường", "lùi_quay_đầu", "vượt_xe", "tai_nạn", "an_toàn", 
        "xe_ưu_tiên", "trừ_điểm", "tước_giấy_phép", "tịch_thu", "giao_nhau", "biển_số"
    ]
    
    # Initialize result structure
    result = []
    
    # Find sections in text
    sections = []
    current_section = None
    current_content = []
    
    # First, split the document into major sections
    lines = text.split('\n')
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Check if line starts a new section (mức phạt)
        if re.match(r'^\d+\.\s+Phạt tiền', line) or "Tịch thu" in line or "Tước quyền" in line or "Trừ" in line or "Biện pháp" in line:
            if current_section:
                sections.append((current_section, current_content))
            current_section = line.strip()
            current_content = []
        elif current_section:
            current_content.append(line.strip())
    
    # Add the last section
    if current_section:
        sections.append((current_section, current_content))
    
    # Process sections into violation types
    violations = {}
    
    for section_title, content in sections:
        # Determine violation type based on content
        assigned_type = None
        
        # Check content against keywords for each violation type
        for violation_type in violation_types:
            keywords = get_keywords_for_type(violation_type)
            for keyword in keywords:
                if keyword in ' '.join(content).lower() or keyword in section_title.lower():
                    if not assigned_type or len(keywords) > len(get_keywords_for_type(assigned_type)):
                        assigned_type = violation_type
        
        # If no specific type found, use a general type
        if not assigned_type:
            assigned_type = "khác"
        
        # Extract fine amount from section title
        fine_amount = section_title
        
        # Create or append to the violation type
        if assigned_type not in violations:
            violations[assigned_type] = []
        
        violation_entry = {
            "mức_phạt": fine_amount,
            "chi_tiết": content
        }
        
        violations[assigned_type].append(violation_entry)
    
    # Build the final result
    for violation_type, entries in violations.items():
        result.append({
            "category": category,
            "violation_type": violation_type,
            "nội_dung": entries
        })
    
    return result

def get_keywords_for_type(violation_type):
    """Return keywords that indicate a particular violation type"""
    keyword_map = {
        "trẻ_em": ["trẻ em", "mầm non", "học sinh", "thiết bị an toàn cho trẻ em"],
        "tốc_độ": ["tốc độ", "quá tốc độ", "chạy quá tốc độ", "dưới tốc độ"],
        "nồng_độ_cồn": ["nồng độ cồn", "cồn"],
        "ma_túy": ["ma túy", "chất kích thích"],
        "đỗ_dừng_xe": ["dừng xe", "đỗ xe", "đậu xe"],
        "giấy_tờ": ["giấy phép", "chứng nhận", "đăng ký xe", "kiểm định", "bảo hiểm"],
        "đường_cao_tốc": ["cao tốc"],
        "biển_báo": ["biển báo", "hiệu lệnh", "đèn tín hiệu"],
        "làn_đường": ["làn đường", "chuyển làn", "phần đường"],
        "chở_người": ["chở người", "chở quá số người", "dây đai an toàn", "buồng lái"],
        "chở_hàng": ["chở hàng", "hàng hóa", "trọng tải", "công-ten-nơ", "rơ moóc"],
        "điện_thoại": ["điện thoại", "thiết bị điện tử"],
        "lái_xe_nguy_hiểm": ["lạng lách", "đánh võng", "chạy đuổi nhau", "dùng chân điều khiển"],
        "kỹ_thuật_xe": ["kính chắn gió", "đèn chiếu sáng", "còi", "gương", "lốp", "phanh"],
        "môi_trường": ["môi trường", "vệ sinh", "rác", "phế thải", "rơi vãi"],
        "lùi_quay_đầu": ["lùi xe", "quay đầu"],
        "vượt_xe": ["vượt xe"],
        "tai_nạn": ["tai nạn", "người bị nạn", "hiện trường"],
        "an_toàn": ["an toàn", "khoảng cách", "còi", "đèn chiếu xa", "không quan sát", "chuyển hướng"],
        "xe_ưu_tiên": ["ưu tiên", "tín hiệu ưu tiên"],
        "biển_số": ["biển số", "gắn biển số"],
        "giao_nhau": ["giao nhau", "đường giao nhau", "đường bộ giao nhau"],
        "trừ_điểm": ["trừ điểm", "điểm giấy phép lái xe"],
        "tước_giấy_phép": ["tước quyền sử dụng giấy phép"],
        "tịch_thu": ["tịch thu"]
    }
    
    return keyword_map.get(violation_type, [])

def cleanup_violation_details(violations):
    """Clean up and format violation details"""
    for violation in violations:
        for entry in violation["nội_dung"]:
            # Clean up details - separate items and remove duplicates
            details = []
            for item in entry["chi_tiết"]:
                # Split by semicolons, commas, and periods
                sub_items = re.split(r'[;,.]\s+', item)
                for sub_item in sub_items:
                    if sub_item and len(sub_item) > 10:  # Avoid very short fragments
                        details.append(sub_item.strip())
            
            # Remove duplicates while preserving order
            seen = set()
            unique_details = []
            for item in details:
                if item not in seen:
                    seen.add(item)
                    unique_details.append(item)
            
            entry["chi_tiết"] = unique_details
    
    return violations

def main(text_file_path, output_file_path):
    # Read input text
    with open(text_file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Parse violations
    violations = parse_traffic_violations(text)
    
    # Clean up details
    violations = cleanup_violation_details(violations)
    
    # Write the result to a JSON file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(violations, file, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete. Output saved to {output_file_path}")

if __name__ == "__main__":
    # Replace with your input and output file paths
    main("/home/ltnga/LawVN-Instructction-Gen/src/data/car.txt", "car_v2.json")