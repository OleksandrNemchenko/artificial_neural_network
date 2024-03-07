
import argparse
import os
import shutil
import subprocess

silence = True
script_path = ""
opencl_code_path = ""

def replace(replacement_str : str, code_source_path : str):
    if not silence : print(f"- Emplace {os.path.basename(code_source_path)}")
    
    args = [
        "python", script_path,
        "--source_pattern_file", opencl_code_path,
        "--source_pattern", replacement_str,
        "--source_code_file", code_source_path,
        "--destination_file", opencl_code_path,
        "--source_code_prefix",  "// OPENCL CODE BEGINNING",
        "--source_code_postfix", "// OPENCL CODE ENDING"
    ]
    
    if silence: args.append("--silence")

    subprocess.run(args)

try:
    parser = argparse.ArgumentParser(description="Make OpenCL code")
    parser.add_argument("--script_path", help="script path")
    parser.add_argument("--opencl_dir", help="directory with OpenCL code")
    parser.add_argument("--ann_meta_configurations_path", help="annMetaConfigurations.hpp path")
    parser.add_argument("--silence", action="store_true", help="no output")
    args = parser.parse_args()

    if not all ([args.script_path, args.opencl_dir]):
        parser.print_help()
    else:
        silence = args.silence
        script_path = args.script_path
        
        if not silence : print(f"- Make openClCode.cpp")

        opencl_code_path = args.opencl_dir + "/" + "openClCode.cpp"
        shutil.copy(args.opencl_dir + "/" + "openClCodePattern.cpp", opencl_code_path)

        replace("REPLACEMENT_ANN_META_CONFIGURATIONS", args.ann_meta_configurations_path)
        replace("REPLACEMENT_UTILITIES_NOT_INTERNAL",  args.opencl_dir + "/../include/artificial_neural_network/utilities.hpp")
        replace("REPLACEMENT_UTILITIES_INTERNALS",     args.opencl_dir + "/../includeInternal/artificial_neural_network_internal/utilities.hpp")
        replace("REPLACEMENT_OPENCL_PROGRAM_HEADER",   args.opencl_dir + "/" + "openClCodeSrc.hpp")
        replace("REPLACEMENT_OPENCL_PROGRAM_CODE",     args.opencl_dir + "/" + "openClCodeSrc.cpp")

except Exception as err:
    print(f"*** Error occured: {err}")
