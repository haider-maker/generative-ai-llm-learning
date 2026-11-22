import subprocess

root = r"E:\GenAI and LLM"

print(f"\nFolder structure of: {root}\n")

subprocess.run(["tree", root, "/F", "/A"], shell=True)