import matplotlib.pyplot as plt
import os


def plot_training_log(train_losses, val_losses, val_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存图表为 PNG 文件
    plt.savefig(os.path.join(output_dir, 'training_plot.png'))
    plt.show()


# 给定的训练日志数据
train_losses = [0.7443104774835358, 0.54122248868075, 0.4553358302566434, 0.3830856370094028, 0.32622450391172086]
val_losses = [0.5773049733391311, 0.5115192898685692, 0.5052643271632603, 0.50756711426883, 0.5485753550069925]
val_accuracies = [0.7594728171334432, 0.7915980230642504, 0.8027182866556837, 0.8095826468973092, 0.8013454146073586]
output_dir = "output_directory"  # 修改为你想要保存图表的目录

# 生成图表并保存为 PNG 文件
plot_training_log(train_losses, val_losses, val_accuracies, output_dir)
