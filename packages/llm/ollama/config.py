import shutil

shutil.copyfile("/etc/nv_tegra_release", "./packages/llm/ollama/nv_tegra_release")
print("##### /etc/nv_tegra_release file copied!")
