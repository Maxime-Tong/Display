import pandas as pd

# 读取CSV文件
df = pd.read_csv('/data/xthuang/workspace/color_display/outputs/display/power_loss_5/DIV2K/metrics.csv')

mean_value = df['psnr'].mean()
print(f"均值: {mean_value}")
