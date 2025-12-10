"""
Data upload utilities for Chameleon CLI.

Handles:
- File path input and validation
- Copying files to project original_data folder
- XLS/XLSX to CSV conversion
- Data schema validation
- Summary display
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from chameleon.cli.validators import (
    get_yes_no,
    validate_file_path,
    ValidationError,
)
from chameleon.cli.file_utils import (
    convert_excel_to_csv,
    detect_file_format,
    load_data_file,
    validate_data_schema,
    standardize_distortion_data,
)


# Expected columns for different data types
EXPECTED_COLUMNS = {
    "original_data": {
        "required": ["question_id", "question"],
        "optional": ["answer_options", "correct_answer", "subject", "topic"],
        "aliases": {
            # Question ID variations
            "q_id": "question_id",
            "id": "question_id",
            "qid": "question_id",
            "question_number": "question_id",
            # Question text variations
            "original_q": "question",
            "original_question": "question",
            "question_text": "question",
            "questiontext": "question",
            "text": "question",
            # Answer options variations
            "ans_options": "answer_options",
            "options": "answer_options",
            "options_json": "answer_options",
            "choices": "answer_options",
            "answers": "answer_options",
            # Correct answer variations
            "correct_ans": "correct_answer",
            "answer": "correct_answer",
            "correct": "correct_answer",
            "solution": "correct_answer",
            # Subject variations
            "category": "subject",
            "domain": "subject",
            "specialty": "subject",
        }
    },
    "distorted_data": {
        "required": ["question_id", "miu", "distorted_question"],
        "optional": ["original_question", "distortion_index", "answer_options", "correct_answer"],
        "aliases": {
            "q_id": "question_id",
            "distorted_q": "distorted_question",
        }
    }
}


def prompt_for_files() -> List[Path]:
    """
    Prompt user for file paths or folder to upload.
    
    Supports:
    - Single file path
    - Folder path (will scan for data files)
    - Multiple comma-separated paths
    - Drag-and-drop (paths with quotes)
    - 'done' to finish
    
    Returns:
        List of validated file paths
    """
    files = []
    supported_extensions = ['.csv', '.xlsx', '.xls', '.json', '.jsonl']
    
    print("\n" + "─" * 50)
    print("📁 STEP 7: Upload Your Data Files")
    print("─" * 50)
    print("""
   Enter path to your data files OR a folder containing them.
   
   Supported formats: CSV, XLSX, XLS, JSON, JSONL
   
   Tips:
   • Enter a FOLDER path to add all data files from it
   • Or enter individual file paths
   • Drag and drop supported
   • XLS/XLSX files will be automatically converted to CSV
   • Type 'done' when finished, or 'skip' to add files later
    """)
    
    while True:
        try:
            user_input = input("   Path (file or folder) or 'done'/'skip': ").strip()
            
            # Handle done/skip
            if user_input.lower() in ['done', 'skip', 'q', 'quit']:
                if user_input.lower() == 'skip':
                    print("   ⏭️ Skipping file upload. Add files manually to original_data/ later.")
                    return []
                break
            
            if not user_input:
                continue
            
            # Clean up path (remove quotes)
            clean_path = user_input.strip().strip('"').strip("'")
            path = Path(clean_path)
            
            # Check if it's a folder
            if path.is_dir():
                print(f"\n   📂 Scanning folder: {path.name}")
                folder_files = []
                
                for ext in supported_extensions:
                    folder_files.extend(path.glob(f"*{ext}"))
                    folder_files.extend(path.glob(f"*{ext.upper()}"))
                
                # Remove duplicates
                folder_files = list(set(folder_files))
                
                if not folder_files:
                    print(f"   ⚠️ No data files found in folder")
                    print(f"      Supported formats: {', '.join(supported_extensions)}")
                else:
                    print(f"   Found {len(folder_files)} data files:")
                    for f in sorted(folder_files)[:10]:
                        print(f"      • {f.name}")
                    if len(folder_files) > 10:
                        print(f"      ... and {len(folder_files) - 10} more")
                    
                    add_all = get_yes_no(f"\n   Add all {len(folder_files)} files?", default=True)
                    
                    if add_all:
                        for f in folder_files:
                            if f not in files:
                                files.append(f)
                        print(f"   ✓ Added {len(folder_files)} files from folder")
            
            elif path.is_file():
                # Single file
                try:
                    validated_path = validate_file_path(
                        str(path), 
                        must_exist=True,
                        extensions=supported_extensions
                    )
                    
                    if validated_path in files:
                        print(f"   ⚠️ Already added: {validated_path.name}")
                    else:
                        files.append(validated_path)
                        print(f"   ✓ Added: {validated_path.name}")
                        
                except ValidationError as e:
                    print(f"   ❌ {e}")
            
            else:
                # Try parsing as multiple comma-separated paths
                paths = parse_file_paths(user_input)
                
                for path_str in paths:
                    try:
                        validated_path = validate_file_path(
                            path_str, 
                            must_exist=True,
                            extensions=supported_extensions
                        )
                        
                        if validated_path in files:
                            print(f"   ⚠️ Already added: {validated_path.name}")
                        else:
                            files.append(validated_path)
                            print(f"   ✓ Added: {validated_path.name}")
                            
                    except ValidationError as e:
                        print(f"   ❌ {e}")
            
            if files:
                print(f"\n   📊 Total files: {len(files)}")
                add_more = get_yes_no("   Add more files/folders?", default=False)
                if not add_more:
                    break
                    
        except KeyboardInterrupt:
            print("\n   Cancelled.")
            break
    
    return files


def parse_file_paths(input_str: str) -> List[str]:
    """
    Parse file paths from user input.
    
    Handles:
    - Quoted paths with spaces
    - Comma-separated paths
    - Mixed quotes and commas
    """
    paths = []
    
    # Remove surrounding quotes if present
    input_str = input_str.strip()
    
    # If the entire input is quoted, treat as single path
    if (input_str.startswith('"') and input_str.endswith('"')) or \
       (input_str.startswith("'") and input_str.endswith("'")):
        return [input_str[1:-1]]
    
    # Split by comma, but respect quotes
    current_path = ""
    in_quotes = False
    quote_char = None
    
    for char in input_str:
        if char in '"\'':
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
            else:
                current_path += char
        elif char == ',' and not in_quotes:
            if current_path.strip():
                paths.append(current_path.strip().strip('"\''))
            current_path = ""
        else:
            current_path += char
    
    if current_path.strip():
        paths.append(current_path.strip().strip('"\''))
    
    return paths


def standardize_columns(df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    """
    Standardize column names and add missing required columns.
    
    Args:
        df: Input DataFrame
        source_name: Source file name for generating IDs
    
    Returns:
        DataFrame with standardized columns
    """
    # Column name mappings (case-insensitive)
    column_map = {
        # Question ID
        'q_id': 'question_id',
        'id': 'question_id',
        'qid': 'question_id',
        'question_number': 'question_id',
        # Question text
        'question_text': 'question',
        'questiontext': 'question',
        'original_q': 'question',
        'original_question': 'question',
        'text': 'question',
        # Answer options
        'options_json': 'answer_options',
        'ans_options': 'answer_options',
        'options': 'answer_options',
        'choices': 'answer_options',
        # Correct answer
        'answer': 'correct_answer',
        'correct_ans': 'correct_answer',
        'correct': 'correct_answer',
        'solution': 'correct_answer',
        # Subject
        'category': 'subject',
        'domain': 'subject',
        'specialty': 'subject',
    }
    
    # Rename columns (case-insensitive matching)
    new_columns = {}
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_").replace("-", "_")
        if col_lower in column_map:
            new_columns[col] = column_map[col_lower]
        elif col_lower in ['question', 'question_id', 'answer_options', 'correct_answer', 'subject', 'topic']:
            new_columns[col] = col_lower
    
    if new_columns:
        df = df.rename(columns=new_columns)
    
    # Auto-generate question_id if missing
    if 'question_id' not in df.columns:
        # Use source file prefix + row number
        prefix = source_name.replace('.csv', '').replace('.xlsx', '').replace(' ', '_')[:10]
        df['question_id'] = [f"{prefix}_{i+1:04d}" for i in range(len(df))]
    
    return df


def copy_files_to_project(
    files: List[Path],
    project_dir: Path,
    convert_excel: bool = True,
    standardize: bool = True
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Copy files to project's original_data folder.
    
    Args:
        files: List of source file paths
        project_dir: Project directory
        convert_excel: Convert XLS/XLSX to CSV
        standardize: Standardize column names
    
    Returns:
        Tuple of (file_info_list, errors)
    """
    original_data_dir = project_dir / "original_data"
    original_data_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    errors = []
    
    print(f"\n   📂 Copying files to {original_data_dir}...")
    
    for source_path in files:
        try:
            format_type = detect_file_format(source_path)
            target_name = source_path.stem + ".csv"
            target_path = original_data_dir / target_name
            
            if format_type == 'excel':
                df = pd.read_excel(source_path)
            else:
                df = load_data_file(source_path)
            
            # Standardize column names
            if standardize:
                df = standardize_columns(df, source_path.name)
            
            # Save as CSV
            df.to_csv(target_path, index=False, encoding='utf-8')
            
            file_info = {
                "source": str(source_path),
                "target": str(target_path),
                "name": target_name,
                "converted_from": source_path.suffix if format_type == 'excel' else None,
                "rows": len(df),
                "columns": list(df.columns),
            }
            
            if format_type == 'excel':
                print(f"   ✓ Converted & standardized: {source_path.name} → {target_name} ({len(df)} rows)")
            else:
                print(f"   ✓ Copied & standardized: {source_path.name} ({len(df)} rows)")
            
            copied_files.append(file_info)
            
        except Exception as e:
            error_msg = f"Failed to copy {source_path.name}: {e}"
            errors.append(error_msg)
            print(f"   ❌ {error_msg}")
    
    return copied_files, errors


def data_sanity_check(df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
    """
    Perform row-by-row sanity check on data.
    
    Checks:
    - Empty question text
    - Empty/invalid answer_options JSON
    - Empty correct_answer
    - Invalid JSON format
    - Duplicate question_ids
    
    Returns:
        Dict with bad_rows list and summary
    """
    import json
    
    bad_rows = []
    
    # Find the actual column names (handle case variations)
    q_col = None
    opt_col = None
    ans_col = None
    id_col = None
    
    for col in df.columns:
        cl = col.lower()
        if cl in ['question', 'question_text', 'text']:
            q_col = col
        elif cl in ['answer_options', 'options_json', 'options']:
            opt_col = col
        elif cl in ['correct_answer', 'answer', 'correct']:
            ans_col = col
        elif cl in ['question_id', 'id', 'q_id']:
            id_col = col
    
    # Check each row
    for idx, row in df.iterrows():
        row_issues = []
        row_num = idx + 2  # +2 for 1-based + header row
        
        # Check question text
        if q_col:
            q_val = row.get(q_col, '')
            if pd.isna(q_val) or str(q_val).strip() == '':
                row_issues.append("Empty question text")
            elif len(str(q_val).strip()) < 10:
                row_issues.append(f"Question too short ({len(str(q_val).strip())} chars)")
        
        # Check answer options
        if opt_col:
            opt_val = row.get(opt_col, '')
            if pd.isna(opt_val) or str(opt_val).strip() == '':
                row_issues.append("Empty answer_options")
            else:
                try:
                    parsed = json.loads(str(opt_val))
                    if not isinstance(parsed, dict):
                        row_issues.append("answer_options not a dict")
                    elif len(parsed) < 2:
                        row_issues.append(f"Only {len(parsed)} option(s)")
                    else:
                        # Check for empty option values
                        empty_opts = [k for k, v in parsed.items() if not v or str(v).strip() == '']
                        if empty_opts:
                            row_issues.append(f"Empty options: {empty_opts}")
                except json.JSONDecodeError as e:
                    row_issues.append(f"Invalid JSON: {str(e)[:30]}")
        
        # Check correct answer
        if ans_col:
            ans_val = row.get(ans_col, '')
            if pd.isna(ans_val) or str(ans_val).strip() == '':
                row_issues.append("Empty correct_answer")
            else:
                # Check if answer is valid (A, B, C, D, or combinations)
                ans_str = str(ans_val).strip().upper()
                valid_letters = set('ABCDEFGH')
                ans_letters = set(c for c in ans_str if c.isalpha())
                if ans_letters and not ans_letters.issubset(valid_letters):
                    row_issues.append(f"Invalid answer: {ans_val}")
        
        if row_issues:
            q_id = row.get(id_col, f"row_{idx}") if id_col else f"row_{idx}"
            bad_rows.append({
                "row": row_num,
                "question_id": q_id,
                "issues": row_issues,
                "question_preview": str(row.get(q_col, ''))[:50] + "..." if q_col else ""
            })
    
    # Check for duplicate question_ids
    if id_col and id_col in df.columns:
        duplicates = df[df.duplicated(subset=[id_col], keep=False)]
        if len(duplicates) > 0:
            dup_ids = duplicates[id_col].unique().tolist()
            for dup_id in dup_ids[:5]:  # Show first 5 duplicates
                dup_rows = df[df[id_col] == dup_id].index.tolist()
                bad_rows.append({
                    "row": [r + 2 for r in dup_rows],
                    "question_id": dup_id,
                    "issues": [f"Duplicate question_id (rows: {[r+2 for r in dup_rows]})"],
                    "question_preview": ""
                })
    
    return {
        "file": file_name,
        "total_rows": len(df),
        "bad_rows": bad_rows,
        "bad_count": len(bad_rows),
        "valid": len(bad_rows) == 0
    }


def display_sanity_check_results(results: List[Dict[str, Any]], project_dir: Path = None) -> bool:
    """
    Display sanity check results to user with detailed explanations and options.
    
    Returns:
        True if all files are valid (or user chose to delete bad rows)
    """
    all_valid = True
    total_bad = 0
    all_bad_rows = []  # Collect all bad rows for potential deletion
    
    print("\n" + "═" * 60)
    print("🔍 DATA SANITY CHECK RESULTS")
    print("═" * 60)
    
    for result in results:
        file_name = result["file"]
        total = result["total_rows"]
        bad_count = result["bad_count"]
        
        if result["valid"]:
            print(f"\n✅ {file_name}: {total} rows - ALL VALID")
        else:
            all_valid = False
            total_bad += bad_count
            print(f"\n❌ {file_name}: {total} rows - {bad_count} ISSUES FOUND")
            print("-" * 50)
            
            # Show ALL bad rows with detailed explanation
            for bad_row in result["bad_rows"]:
                row_num = bad_row["row"]
                q_id = bad_row["question_id"]
                issues = bad_row["issues"]
                preview = bad_row.get("question_preview", "")
                
                all_bad_rows.append({
                    "file": file_name,
                    "row": row_num,
                    "question_id": q_id,
                    "issues": issues
                })
                
                if isinstance(row_num, list):
                    print(f"\n   📍 Rows {row_num}")
                    print(f"      Question ID: {q_id}")
                else:
                    print(f"\n   📍 Row {row_num}")
                    print(f"      Question ID: {q_id}")
                
                # Detailed issue explanations
                for issue in issues:
                    if "Empty question" in issue:
                        print(f"      ❌ {issue}")
                        print(f"         → The question text is missing or blank")
                        print(f"         💡 Fix: Add the question text, or delete this row")
                    elif "Empty answer_options" in issue:
                        print(f"      ❌ {issue}")
                        print(f"         → The answer choices (A, B, C, D) are missing")
                        print(f"         💡 Fix: Add JSON like {{\"A\": \"...\", \"B\": \"...\", \"C\": \"...\", \"D\": \"...\"}}")
                    elif "Empty correct_answer" in issue:
                        print(f"      ❌ {issue}")
                        print(f"         → The correct answer letter is missing")
                        print(f"         💡 Fix: Add the correct answer (A, B, C, or D)")
                    elif "Invalid JSON" in issue:
                        print(f"      ❌ {issue}")
                        print(f"         → The answer_options column has malformed JSON")
                        print(f"         💡 Fix: Ensure valid JSON format")
                    elif "Duplicate" in issue:
                        print(f"      ❌ {issue}")
                        print(f"         💡 Fix: Remove duplicate entries")
                    elif "too short" in issue.lower():
                        print(f"      ❌ {issue}")
                        print(f"         💡 Fix: Question text should be at least 10 characters")
                    else:
                        print(f"      ❌ {issue}")
                
                if preview:
                    print(f"      📝 Preview: \"{preview}\"")
    
    print("\n" + "═" * 60)
    
    if all_valid:
        print("✅ ALL FILES PASSED SANITY CHECK")
        print("═" * 60)
        return True
    
    print(f"❌ FOUND {total_bad} TOTAL ISSUES")
    print("═" * 60)
    
    # Offer options to user
    if project_dir and all_bad_rows:
        print(f"\n📋 Summary of {total_bad} problematic row(s):")
        for i, bad in enumerate(all_bad_rows, 1):
            print(f"   {i}. {bad['file']} → Row {bad['row']} ({bad['question_id']})")
        
        print("\n" + "─" * 50)
        print("What would you like to do?")
        print("   [D] Delete ALL problematic rows automatically")
        print("   [S] Skip - I'll fix them manually and re-upload")
        print("   [P] Proceed anyway (not recommended)")
        print("─" * 50)
        
        while True:
            choice = input("\n   Your choice (D/S/P): ").strip().upper()
            
            if choice == 'D':
                # Delete all bad rows
                print("\n🗑️  Deleting problematic rows...")
                deleted_count = _delete_bad_rows(project_dir, all_bad_rows)
                print(f"✅ Deleted {deleted_count} row(s). Data is now clean!")
                return True
            
            elif choice == 'S':
                print("\n⏭️  Skipped. Please fix the issues and re-upload.")
                print("   Run: python cli.py init  (and upload fixed files)")
                return False
            
            elif choice == 'P':
                print("\n⚠️  Proceeding with issues. Some questions may fail during processing.")
                return True
            
            else:
                print("   Please enter D, S, or P")
    
    return all_valid


def _delete_bad_rows(project_dir: Path, bad_rows: List[Dict]) -> int:
    """Delete bad rows from CSV files."""
    import pandas as pd
    
    # Group by file
    files_to_fix = {}
    for bad in bad_rows:
        file_name = bad["file"]
        row_num = bad["row"]
        
        if file_name not in files_to_fix:
            files_to_fix[file_name] = []
        
        # Convert row number to index (row_num is 1-based + header, so index = row_num - 2)
        if isinstance(row_num, list):
            for r in row_num:
                files_to_fix[file_name].append(r - 2)
        else:
            files_to_fix[file_name].append(row_num - 2)
    
    deleted_total = 0
    
    for file_name, indices in files_to_fix.items():
        file_path = project_dir / "original_data" / file_name
        
        if file_path.exists():
            df = pd.read_csv(file_path, encoding='utf-8')
            original_len = len(df)
            
            # Remove duplicates from indices and filter valid ones
            indices = list(set(i for i in indices if 0 <= i < len(df)))
            
            if indices:
                df = df.drop(indices)
                df.to_csv(file_path, index=False, encoding='utf-8')
                deleted = original_len - len(df)
                deleted_total += deleted
                print(f"   ✓ {file_name}: deleted {deleted} row(s)")
    
    return deleted_total


def validate_uploaded_data(project_dir: Path, show_bad_rows: bool = True) -> Dict[str, Any]:
    """
    Validate uploaded data files against expected schema with row-level sanity check.
    
    Args:
        project_dir: Project directory
        show_bad_rows: Whether to show detailed row-level issues
    
    Returns:
        Validation report
    """
    original_data_dir = project_dir / "original_data"
    report = {
        "valid": True,
        "files": [],
        "warnings": [],
        "errors": [],
        "column_mappings": {},
        "sanity_results": [],
    }
    
    # Find all data files
    data_files = list(original_data_dir.glob("*.csv")) + \
                 list(original_data_dir.glob("*.json")) + \
                 list(original_data_dir.glob("*.jsonl"))
    
    if not data_files:
        report["valid"] = False
        report["errors"].append("No data files found in original_data/")
        return report
    
    expected = EXPECTED_COLUMNS["original_data"]
    
    # Build case-insensitive alias map
    alias_map = {}
    for alias, standard in expected["aliases"].items():
        alias_map[alias.lower()] = standard
    
    for file_path in data_files:
        file_report = {
            "name": file_path.name,
            "valid": True,
            "rows": 0,
            "columns": [],
            "missing_required": [],
            "missing_optional": [],
            "detected_mappings": {},
        }
        
        try:
            df = load_data_file(file_path)
            file_report["rows"] = len(df)
            file_report["columns"] = list(df.columns)
            
            # Apply column aliases (case-insensitive)
            mapped_columns = set()
            
            for col in df.columns:
                col_lower = col.lower().replace(" ", "_").replace("-", "_")
                
                if col_lower in alias_map:
                    mapped_columns.add(alias_map[col_lower])
                    file_report["detected_mappings"][col] = alias_map[col_lower]
                elif col.lower() in [r.lower() for r in expected["required"]]:
                    # Direct match (case insensitive)
                    for req in expected["required"]:
                        if col.lower() == req.lower():
                            mapped_columns.add(req)
                            break
                elif col.lower() in [o.lower() for o in expected["optional"]]:
                    # Direct match for optional
                    for opt in expected["optional"]:
                        if col.lower() == opt.lower():
                            mapped_columns.add(opt)
                            break
                else:
                    mapped_columns.add(col)
            
            # Check required columns
            for req_col in expected["required"]:
                if req_col not in mapped_columns:
                    file_report["missing_required"].append(req_col)
                    file_report["valid"] = False
            
            # Check optional columns
            for opt_col in expected["optional"]:
                if opt_col not in mapped_columns:
                    file_report["missing_optional"].append(opt_col)
            
            if not file_report["valid"]:
                report["valid"] = False
                report["errors"].append(
                    f"{file_path.name}: Missing required columns: {file_report['missing_required']}"
                )
            
            # Store detected mappings
            if file_report["detected_mappings"]:
                report["column_mappings"][file_path.name] = file_report["detected_mappings"]
            
            # Row-level sanity check
            if show_bad_rows:
                sanity_result = data_sanity_check(df, file_path.name)
                report["sanity_results"].append(sanity_result)
                
                if not sanity_result["valid"]:
                    report["valid"] = False
                    report["warnings"].append(
                        f"{file_path.name}: {sanity_result['bad_count']} rows have issues"
                    )
                
        except Exception as e:
            file_report["valid"] = False
            report["valid"] = False
            report["errors"].append(f"{file_path.name}: {e}")
        
        report["files"].append(file_report)
    
    # Display sanity check results if enabled (pass project_dir for deletion option)
    if show_bad_rows and report["sanity_results"]:
        sanity_passed = display_sanity_check_results(report["sanity_results"], project_dir)
        if sanity_passed:
            report["valid"] = True  # User chose to delete bad rows or proceed
            report["warnings"] = []  # Clear warnings since they're resolved
    
    return report


def display_upload_summary(
    copied_files: List[Dict[str, Any]],
    validation_report: Dict[str, Any]
) -> None:
    """Display summary of uploaded files and validation."""
    
    print("\n" + "═" * 50)
    print("📊 DATA UPLOAD SUMMARY")
    print("═" * 50)
    
    if not copied_files:
        print("\n   No files uploaded.")
        return
    
    print(f"\n   Files uploaded: {len(copied_files)}")
    
    total_rows = sum(f.get("rows", 0) for f in copied_files)
    print(f"   Total rows: {total_rows}")
    
    print("\n   Files:")
    for f in copied_files:
        converted = f" (converted from {f.get('converted_from', '')})" if f.get('converted_from') else ""
        print(f"   • {f['name']}: {f['rows']} rows, {len(f['columns'])} columns{converted}")
    
    # Validation results
    if validation_report["valid"]:
        print("\n   ✅ Data validation: PASSED")
    else:
        print("\n   ❌ Data validation: FAILED")
        for error in validation_report["errors"]:
            print(f"      • {error}")
    
    if validation_report["warnings"]:
        print("\n   ⚠️ Warnings:")
        for warning in validation_report["warnings"]:
            print(f"      • {warning}")
    
    # Column info
    print("\n   Detected columns:")
    all_columns = set()
    for f in copied_files:
        all_columns.update(f.get("columns", []))
    
    for col in sorted(all_columns):
        print(f"      • {col}")


def run_file_upload_flow(project_dir: Path) -> bool:
    """
    Run the complete file upload flow.
    
    Args:
        project_dir: Project directory path
    
    Returns:
        True if files were uploaded and valid, False otherwise
    """
    # Prompt for files
    files = prompt_for_files()
    
    if not files:
        return False
    
    # Copy files
    copied_files, copy_errors = copy_files_to_project(files, project_dir)
    
    if not copied_files:
        print("\n   ❌ No files were copied successfully.")
        return False
    
    # Validate
    validation_report = validate_uploaded_data(project_dir)
    
    # Display summary
    display_upload_summary(copied_files, validation_report)
    
    # Ask if user wants to proceed
    if validation_report["valid"]:
        return True
    else:
        proceed = get_yes_no("\n   Proceed despite validation errors?", default=False)
        return proceed

