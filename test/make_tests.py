
from pathlib import Path
import argparse
import shutil
import subprocess

silence = False
directory = Path()
script_path = Path()

def process_file(
        source_pattern_file: Path,
        source_pattern: str,
        source_code_file: Path,
        destination_file: Path,
        source_code_prefix: str = None,
        source_code_postfix: str = None
):

    args = ["python", script_path,
        "--source_pattern_file", directory + "/" + source_pattern_file,
        "--source_pattern",      source_pattern,
        "--source_code_file",    directory + "/" + source_code_file,
        "--destination_file",    directory + "/" + destination_file
    ]

    if (source_code_prefix):
        args.append("--source_code_prefix")
        args.append(source_code_prefix)

    if (source_code_postfix):
        args.append("--source_code_postfix")
        args.append(source_code_postfix)

    if (silence):
        args.append("--silence")
    
    subprocess.run(args)

if __name__ == "__main__":
    
    try:
        parser = argparse.ArgumentParser(description="Make test sequences")
        parser.add_argument("--script_path", help="script path")
        parser.add_argument("--dir", help="directory with patterns, input and output files")
        parser.add_argument("--silence", action="store_true", help="no output")
        args = parser.parse_args()

        if not all ([args.script_path, args.dir]):
            parser.print_help()
        else:
            silence = args.silence
            directory = args.dir
            script_path = args.script_path

            if not silence : print(f"- Working directory: {args.dir}")

            shutil.copy(directory + "/" + "testsPattern.cpp", directory + "/" + "tests.cpp")
            
            process_file("tests.cpp", "FEED_FORWARD_1_JSON", "feedForward1.json", "tests.cpp" )
            process_file("tests.cpp", "FEED_FORWARD_2_JSON", "feedForward2.json", "tests.cpp" )

    except Exception as err:
        print(f"*** Error occured: {err}")
        