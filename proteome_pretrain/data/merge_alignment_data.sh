#!/bin/bash
# 合并分割的alignment_data.txt文件

echo "正在合并 alignment_data.txt 文件..."
cat alignment_data.txt.part_* > alignment_data.txt

if [ -f "alignment_data.txt" ]; then
    echo "✓ 合并完成: alignment_data.txt"
    echo "文件大小:"
    ls -lh alignment_data.txt
    echo ""
    echo "您可以删除分割文件以节省空间:"
    echo "rm alignment_data.txt.part_*"
else
    echo "✗ 合并失败"
    exit 1
fi

