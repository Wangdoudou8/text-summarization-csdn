# text-summarization-csdn
 An open source project on my CSDN blog, whose dataset is the CNN/DM and whose model is T5.



详细的项目使用说明看见博客：https://blog.csdn.net/Wang_Dou_Dou_/article/details/129544380



我是基于 T5(text-to-text-transfer-transformer)模型的源代码修改的，地址：https://github.com/Shivanandroy/simpleT5。主要修改点如下：

  1. 取消了 checkpoint 保存机制，只保存最后一轮的模型和训练日志
  2. 加入早停机制，使得模型在训练过程中，若发现 val_loss(验证集的损失) 没有下降，就及时停止训练(以防止过拟合)。
  3. 使用了 CNN/Daily Mail 的报刊新闻(一部分) 作为我的数据集，train.csv 有 9000 个样本(我将其以 9:1 的形式划分成了训练集和验证集)，无测试集。
    4. 加入了 ROUGE 指标，主要在模型训练完后，对验证集进行 ROUGE 分数计算。
    5. 将所有重要的英文注释翻译为中文，并加入详细的注释。

