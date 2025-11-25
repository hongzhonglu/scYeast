#!/bin/bash
# 合并分割的pro_interaction.pkl文件

echo "正在合并 pro_interaction.pkl 文件..."
cat pro_interaction.pkl.part_* > pro_interaction.pkl

if [ -f "pro_interaction.pkl" ]; then
    echo "✓ 合并完成: pro_interaction.pkl"
    echo "文件大小:"
    ls -lh pro_interaction.pkl
    echo ""
    echo "您可以删除分割文件以节省空间:"
    echo "rm pro_interaction.pkl.part_*"
else
    echo "✗ 合并失败"
    exit 1
fi

