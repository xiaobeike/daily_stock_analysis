# -*- coding: utf-8 -*-
"""
===================================
清理无效股票名称缓存
===================================

运行此脚本清理数据库中保存的无效股票名称，如 "股票000333"、纯数字等

使用方法：
    python cleanup_stock_names.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from storage import get_db


def is_valid_stock_name(code: str, name: str) -> bool:
    """
    验证股票名称是否有效（复制自 main.py）
    """
    if not name:
        return False

    name = name.strip()

    # 长度检查
    if len(name) < 2 or len(name) > 10:
        return False

    # 不能与代码相同
    if name == code:
        return False

    # 不能以"股票"开头
    if name.startswith("股票"):
        return False

    # 不能全是数字
    if name.replace(".", "").replace("-", "").isdigit():
        return False

    # 必须包含汉字或字母
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in name)
    has_letter = any('a' <= c.lower() <= 'z' for c in name)
    if not has_chinese and not has_letter:
        return False

    return True


def main():
    """主函数：清理无效股票名称缓存"""
    print("=" * 60)
    print("清理无效股票名称缓存")
    print("=" * 60)

    db = get_db()

    # 获取所有缓存的股票名称
    from storage import StockNameCache
    from storage import select

    with db.get_session() as session:
        all_records = session.execute(
            select(StockNameCache)
        ).scalars().all()

        print(f"\n当前数据库中共有 {len(all_records)} 条股票名称缓存\n")

        deleted_count = 0
        invalid_names = []

        for record in all_records:
            code = record.code
            name = record.name

            if not is_valid_stock_name(code, name):
                reason = ""
                if not name or not name.strip():
                    reason = "空字符串"
                elif name.startswith("股票"):
                    reason = "以'股票'开头"
                elif name == code:
                    reason = "与代码相同"
                elif name.replace(".", "").replace("-", "").isdigit():
                    reason = "纯数字"
                elif len(name.strip()) < 2 or len(name.strip()) > 10:
                    reason = f"长度不合法({len(name)}字符)"
                else:
                    reason = "未知原因"

                invalid_names.append((code, name, reason))
                session.delete(record)
                deleted_count += 1

        if invalid_names:
            session.commit()
            print(f"已删除 {deleted_count} 条无效记录:\n")
            for code, name, reason in invalid_names:
                print(f"  ❌ {code}: '{name}' ({reason})")
        else:
            print("✅ 没有发现无效的股票名称缓存")

        # 显示有效的缓存记录
        valid_records = [r for r in all_records if is_valid_stock_name(r.code, r.name)]
        if valid_records:
            print(f"\n有效缓存 ({len(valid_records)} 条):")
            for r in valid_records[:10]:  # 只显示前10条
                print(f"  ✅ {r.code}: '{r.name}'")
            if len(valid_records) > 10:
                print(f"  ... 还有 {len(valid_records) - 10} 条")

    print("\n" + "=" * 60)
    print("清理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
