import json
import os

def add_category_to_lines(data):
    """
    Add the category name in parentheses to the end of each line in the nội_dung lists
    if it's not already present.
    """
    for category_data in data:
        # Get the category name
        category = category_data["category"]
        suffix = f"({category})"
        
        # Process each item in the nội_dung list
        for i, line in enumerate(category_data["nội_dung"]):
            # Check if the line already ends with the suffix
            if not line.endswith(suffix):
                # Remove any existing parentheses at the end if present
                if line.endswith(")"):
                    last_open_paren = line.rfind("(")
                    if last_open_paren != -1:
                        line = line[:last_open_paren].strip()
                
                # Add the category suffix
                category_data["nội_dung"][i] = line + " " + suffix
    
    return data

def main():
    # Ask for input filename
    input_file = input("Enter the input JSON filename: ")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return
    
    try:
        # Read data from input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process the data
        processed_data = add_category_to_lines(data)
        
        # Generate output filename
        filename, ext = os.path.splitext(input_file)
        output_file = f"{filename}_processed{ext}"
        
        # Save processed data to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Processed data has been saved to '{output_file}'")
        
        # Print a sample of the processed data
        print("\nSample of processed data:")
        if isinstance(processed_data, list) and len(processed_data) > 0:
            sample_category = processed_data[0]["category"]
            sample_items = processed_data[0]["nội_dung"][:2]  # First 2 items
            print(f"Category: {sample_category}")
            for item in sample_items:
                print(f"- {item}")
        
    except json.JSONDecodeError:
        print(f"Error: '{input_file}' is not a valid JSON file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()