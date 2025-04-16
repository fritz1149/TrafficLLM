cat $1 | while read line
do
    echo "准备安装：${line}"
    pip install $line
    if [$? -neq 0 ]; then
        echo "安装失败：${line}"
        exit 1
    fi
done
