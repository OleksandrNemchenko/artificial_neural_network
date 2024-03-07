
import argparse

silence = False

def load_text_file(text_file_path: str, format: str = 'utf-8') -> str:
    with open(text_file_path, 'r', encoding=format) as source_file:
        content = source_file.read()
    return content

def write_text_file(destination_file: str, text_file_content: str, format: str = 'utf-8') :
    with open(destination_file, 'w', encoding=format) as dest_file:
        dest_file.write(text_file_content)
        dest_file.flush()

def get_source_code(file_str: str, prefix: str, postfix: str) -> str:
    if prefix is None and postfix is None:
        return file_str
    start_position = file_str.find(prefix) + len(prefix)
    end_position = file_str.find(postfix)
    result = file_str[start_position : end_position]
    return result

def make_file(pattern_file: str, kernel_code: str, source_pattern: str) -> str:
    result = pattern_file.replace(source_pattern, kernel_code)
    return result
    
def prepare_parser():
    parser = argparse.ArgumentParser(description="Make code from pattern and source")
    
    parser.add_argument("--source_pattern_file", help="path to the source file that will be used as pattern")
    parser.add_argument("--source_pattern", help="pattern in the source file to be replaced by code")
    parser.add_argument("--source_code_file", help="path to the source file with the code to be used while replacing pattern in the source C++ file")
    parser.add_argument("--source_code_prefix", help="prefix for code. Used to copy code to the destination file. Can be omitted if full file has to be copied")
    parser.add_argument("--source_code_postfix", help="postfix for code. Used to copy code to the destination file. Can be omitted if full file has to be copied")
    parser.add_argument("--destination_file", help="file to the destination file")
    parser.add_argument("--silence", action="store_true", help="no output")
    
    return parser

if __name__ == "__main__":
    
    try:
        parser = prepare_parser()
        args = parser.parse_args()
        silence = args.silence
            
        if not all ([args.source_pattern_file, args.source_pattern, args.source_code_file, args.destination_file]):
            parser.print_help()
        else:
            if not silence : print(f"- Load source code file {args.source_code_file}")
            kernel_file = load_text_file(args.source_code_file)
            
            if not silence : print(f"- Extract source code inside tags {args.source_code_prefix} .. {args.source_code_postfix}")
            kernel_code = get_source_code(kernel_file, args.source_code_prefix, args.source_code_postfix)
            
            if not silence : print(f"- Load pattern file {args.source_pattern_file}")
            pattern_file = load_text_file(args.source_pattern_file)
            
            if not silence : print(f"- Make destination file")
            destination_file = make_file(pattern_file, kernel_code, args.source_pattern)

            if not silence : print(f"- Save destination file {args.destination_file}")
            write_text_file(args.destination_file, destination_file)
            
    except Exception as err:
        print(f"*** Error occured: {err}")
        