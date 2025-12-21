
from asyncio.log import logger
import json
import config
from pathlib import Path
from stage_first import prepare_stage1
from stage_second import stage2_update_memory_from_dpo
import logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
def main():
    """主函数：命令行入口"""
    logger.info("=" * 60)
    logger.info("MetaEvo Pipeline 启动")
    logger.info("=" * 60)
    
    if config.DEBUG_MODE:
        logger.info("调试模式已启用")
        logger.info(f"配置信息:\n{json.dumps(config.get_config_summary(), indent=2, ensure_ascii=False)}")

    print("\n" + "=" * 60)
    print("MetaEvo Pipeline")
    print("=" * 60)
    print("1. 准备阶段 1 - 生成元优化数据 (Generate DPO Data)")
    print("2. 准备阶段 2 - 使用强化结果更新Memory (Update Memory)")
    print("3. 推理阶段 - 使用Memory进行推理 (Inference)")
    print("=" * 60)
    
    choice = input("请选择阶段 (输入 1 / 2 / 3): ").strip()
    
    # 加载数据集
    input_file = Path(config.original_data_file)
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        print(f"[✘] 错误: 输入文件不存在 - {input_file}")
        return
    
    try:
        logger.info(f"加载数据集: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"数据集加载成功，共 {len(dataset)} 条数据")
    except Exception as e:
        logger.error(f"加载数据集失败: {e}", exc_info=True)
        print(f"[✘] 错误: 无法加载数据集 - {e}")
        return
    
    # 执行对应阶段
    try:
        if choice == "1":
            prepare_stage1(dataset)
        elif choice == "2":
            stage2_update_memory_from_dpo()
        elif choice == "3":
            inference_with_memory(dataset)
        else:
            logger.warning(f"无效的选择: {choice}")
            print("[✘] 无效的选择，请输入 1 / 2 / 3")
            
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        print("\n[!] 程序已中断")
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        print(f"[✘] 程序执行失败: {e}")

if __name__ == "__main__":
    main()