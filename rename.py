import os
import fileinput
import shutil

# 项目根目录路径
PROJECT_ROOT = "/Users/didi/PycharmProject/BACFormer"
# 备份目录路径（在原项目名后添加后缀）
BACKUP_DIR = f"{PROJECT_ROOT}_left_atrium_backup"
# 关键词替换映射表：{旧关键词: 新关键词}
REPLACE_MAP = {
    "LeftAtrium": "LeftAtrium",
    "left_atrium": "left_atrium",
    "LEFT_ATRIUM": "LEFT_ATRIUM"
}
# 需要处理的文件类型列表
FILE_TYPES = [".py", ".md", ".txt", ".yml", ".yaml", ".json"]
# 需要排除的目录列表（不参与遍历）
EXCLUDE_DIRS = ["__pycache__", "venv", "env", "build", "dist", ".git", ".idea"]


def backup_project():
    """
    备份整个项目到指定目录
    
    在执行批量替换前创建项目备份，防止操作失误导致数据丢失
    """
    # 检查备份目录是否已存在
    if not os.path.exists(BACKUP_DIR):
        print(f"📦 正在备份项目到 {BACKUP_DIR}...")
        # 递归复制整个项目目录到备份位置
        shutil.copytree(PROJECT_ROOT, BACKUP_DIR)
        print("✅ 备份完成！")
    else:
        # 如果备份目录已存在，跳过备份步骤
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
        # 使用fileinput以 inplace=True 模式打开文件，直接修改原文件
        with fileinput.FileInput(file_path, inplace=True, encoding="utf-8") as f:
            # 逐行读取文件内容
            for line in f:
                # 初始化新行为原始行
                new_line = line
                # 遍历替换映射表
                for old, new in REPLACE_MAP.items():
                    # 在当前行中替换所有旧关键词为新关键词
                    new_line = new_line.replace(old, new)
                # 输出替换后的行（写入原文件）
                print(new_line, end="")
        return True
    except Exception as e:
        # 捕获异常并打印错误信息
        print(f"❌ 处理文件失败 {file_path}：{str(e)}")
        return False


def rename_files_and_dirs():
    """
    重命名包含旧关键词的文件和目录
    
    自底向上遍历项目目录，重命名所有包含REPLACE_MAP中关键词的文件和文件夹
    """
    # 自底向上遍历项目目录树（topdown=False确保先处理子目录）
    for root, dirs, files in os.walk(PROJECT_ROOT, topdown=False):
        # 过滤掉需要排除的目录
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        # 处理当前目录下的所有文件
        for file in files:
            # 构建文件的完整路径
            old_path = os.path.join(root, file)
            # 初始化新文件名
            new_file = file
            # 遍历替换映射表，替换文件名中的关键词
            for old, new in REPLACE_MAP.items():
                new_file = new_file.replace(old, new)
            # 构建新文件的完整路径
            new_path = os.path.join(root, new_file)
            # 如果新旧路径不同且新路径不存在，执行重命名
            if old_path != new_path and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"📝 重命名文件：{old_path} → {new_path}")

        # 处理当前目录下的所有子目录
        for dir in dirs:
            # 构建目录的完整路径
            old_path = os.path.join(root, dir)
            # 初始化新目录名
            new_dir = dir
            # 遍历替换映射表，替换目录名中的关键词
            for old, new in REPLACE_MAP.items():
                new_dir = new_dir.replace(old, new)
            # 构建新目录的完整路径
            new_path = os.path.join(root, new_dir)
            # 如果新旧路径不同且新路径不存在，执行重命名
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
    # 第一步：备份整个项目
    backup_project()
    print("\n🔍 开始全局替换关键词...")
    
    # 初始化计数器
    success_count = 0  # 成功处理的文件数
    total_count = 0    # 总共需要处理的文件数
    
    # 自上而下遍历项目目录树
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # 过滤掉需要排除的目录
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        # 遍历当前目录下的所有文件
        for file in files:
            # 检查文件扩展名是否在需要处理的类型列表中
            if any(file.endswith(ft) for ft in FILE_TYPES):
                # 总文件数加1
                total_count += 1
                # 构建文件的完整路径
                file_path = os.path.join(root, file)
                # 调用替换函数，如果成功则成功计数加1
                if replace_in_file(file_path):
                    success_count += 1
    
    print("\n📎 开始重命名文件/目录...")
    # 第二步：重命名文件和目录
    rename_files_and_dirs()
    
    # 打印完成信息和统计结果
    print(f"\n🎉 批量替换完成！")
    print(f"📊 统计：共处理 {total_count} 个文件，成功 {success_count} 个")
    print(f"⚠️  若出错，可从 {BACKUP_DIR} 恢复备份")


if __name__ == "__main__":
    # 程序入口：执行主函数
    main()