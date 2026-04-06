import os
import fileinput
import shutil

PROJECT_ROOT = "/Users/didi/PycharmProject/BACFormer"
BACKUP_DIR = f"{PROJECT_ROOT}_left_atrium_backup"
REPLACE_MAP = {
    "LeftAtrium": "LeftAtrium",
    "left_atrium": "left_atrium",
    "LEFT_ATRIUM": "LEFT_ATRIUM"
}
FILE_TYPES = [".py", ".md", ".txt", ".yml", ".yaml", ".json"]
EXCLUDE_DIRS = ["__pycache__", "venv", "env", "build", "dist", ".git", ".idea"]


def backup_project():
    """
    备份整个项目到指定目录
    
    在执行批量替换前创建项目备份，防止操作失误导致数据丢失
    """
    if not os.path.exists(BACKUP_DIR):
        print(f"📦 正在备份项目到 {BACKUP_DIR}...")
        shutil.copytree(PROJECT_ROOT, BACKUP_DIR)
        print("✅ 备份完成！")
    else:
        print("ℹ️ 备份目录已存在，跳过备份")


def replace_in_file(file_path):
    """
    替换单个文件中的关键词
    
    遍历文件内容，将REPLACE_MAP中定义的旧关键词替换为新关键词
    
    Args:
        file_path (str): 需要处理的文件路径
        
    Returns:
        bool: 替换成功返回True，失败返回False
    """
    try:
        with fileinput.FileInput(file_path, inplace=True, encoding="utf-8") as f:
            for line in f:
                new_line = line
                for old, new in REPLACE_MAP.items():
                    new_line = new_line.replace(old, new)
                print(new_line, end="")
        return True
    except Exception as e:
        print(f"❌ 处理文件失败 {file_path}：{str(e)}")
        return False


def rename_files_and_dirs():
    """
    重命名包含旧关键词的文件和目录
    
    自底向上遍历项目目录，重命名所有包含REPLACE_MAP中关键词的文件和文件夹
    """
    for root, dirs, files in os.walk(PROJECT_ROOT, topdown=False):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in files:
            old_path = os.path.join(root, file)
            new_file = file
            for old, new in REPLACE_MAP.items():
                new_file = new_file.replace(old, new)
            new_path = os.path.join(root, new_file)
            if old_path != new_path and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"📝 重命名文件：{old_path} → {new_path}")

        for dir in dirs:
            old_path = os.path.join(root, dir)
            new_dir = dir
            for old, new in REPLACE_MAP.items():
                new_dir = new_dir.replace(old, new)
            new_path = os.path.join(root, new_dir)
            if old_path != new_path and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"📁 重命名目录：{old_path} → {new_path}")


def main():
    """
    主函数：执行项目关键词批量替换流程
    
    1. 先备份项目
    2. 遍历所有指定类型的文件进行内容替换
    3. 重命名包含旧关键词的文件和目录
    """
    backup_project()
    print("\n🔍 开始全局替换关键词...")
    success_count = 0
    total_count = 0
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if any(file.endswith(ft) for ft in FILE_TYPES):
                total_count += 1
                file_path = os.path.join(root, file)
                if replace_in_file(file_path):
                    success_count += 1
    print("\n📎 开始重命名文件/目录...")
    rename_files_and_dirs()
    print(f"\n🎉 批量替换完成！")
    print(f"📊 统计：共处理 {total_count} 个文件，成功 {success_count} 个")
    print(f"⚠️  若出错，可从 {BACKUP_DIR} 恢复备份")


if __name__ == "__main__":
    main()