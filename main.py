from abc import ABC
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.optim import AdamW
import numpy as np
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from rouge import Rouge
from transformers import (
    T5ForConditionalGeneration,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
)


class myDataModule(Dataset):
    """ Pytorch 类型的数据集类 """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            source_max_token_len: int = 512,
            target_max_token_len: int = 512,
    ):
        """
        为输入数据初始化 PyTorch 数据集模块
        参数:
            data (pd.DataFrame): 输入为 pandas dataframe 形式. Dataframe 必须有 2 列 --> "source_text" 和 "target_text"
            tokenizer (PreTrainedTokenizer): 一个预训练好的分词器 (例如 T5Tokenizer, MT5Tokenizer 或 ByT5Tokenizer)
            source_max_token_len (int, optional): 源文本的最大 token 长度. 默认值为 512.
            target_max_token_len (int, optional): 目标文本的最大 token 长度. 默认值为 512.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        """ 返回数据的长度 """
        return len(self.data)

    def __getitem__(self, index: int):
        """ 返回一个 batch_size 的 input, 以便之后输入模型 """
        # 1. 获取数据集中第 index 批次的数据
        data_row = self.data.iloc[index]
        # 2. 获取源文档
        source_text = data_row["source_text"]
        # 3. 对源文档进行分词编码
        source_text_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_token_len,
            padding="max_length",  # 如果文档的长度未达到 max_length(即self.source_max_token_len), 就用<pad>填充满
            truncation=True,  # 如果文档的长度超过 max_length, 则截断后面的文本, 只取前面的
            return_attention_mask=True,  # 要求返回 attention 的掩码张亮
            add_special_tokens=True,  # 要求增加特殊的 token, 比如加入 <pad> 位
            return_tensors="pt",  # 以 pytorch 的 tensor 类型返回
        )
        # 4. 对目标文档(即摘要)进行分词编码
        target_text_encoding = self.tokenizer(
            data_row["target_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # 5. 获取目标文本(即摘要)的 token ids
        labels = target_text_encoding["input_ids"]
        # 6. 将摘要中 token id 为 0(即<pad>位) 的转为 -100, 便于后期 "省略" 掉
        labels[labels == 0] = -100
        # 7. 以字典(dict)的形式返回这一批次的数据
        return dict(
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),  # .flatten() 是为了
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
        )


class myLightningDataModule(pl.LightningDataModule, ABC):
    """ PyTorch Lightning 类型的数据集类, 它继承了 "PyTorch 类型的数据集类" 的所有东西, 并附加了其他功能 """

    def __init__(
            self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            batch_size: int = 4,
            source_max_token_len: int = 512,
            target_max_token_len: int = 512,
            num_workers: int = 4,  # 要为数据使用多少子进程装载。'0'表示将在主进程中加载数据(默认值:'0')
    ):
        """
        初始化 PyTorch Lightning 类型的数据模块
        参数:
            train_df (pd.DataFrame): training dataframe. Dataframe 必须有 2 列 --> "source_text" 和 "target_text"
            test_df (pd.DataFrame): validation dataframe. Dataframe 必须有 2 列 --> "source_text" 和 "target_text"
            tokenizer (PreTrainedTokenizer): 一个预训练好的分词器 (例如 T5Tokenizer, MT5Tokenizer 或 ByT5Tokenizer)
            batch_size (int, optional): batch size. 默认为 4.
            source_max_token_len (int, optional): 源文本的最大 token 长度. 默认值为 512.
            target_max_token_len (int, optional): 目标文本的最大 token 长度. 默认值为 512.
        """
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.num_workers = num_workers
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        self.train_dataset = myDataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.test_dataset = myDataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        """ 训练集 dataloader """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 随机打乱训练集
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """ 验证集 dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """ 测试集 dataloader """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class myLightningModel(pl.LightningModule, ABC):
    """ PyTorch Lightning 模型类 """

    def __init__(
            self,
            tokenizer,
            model,
            output_dir: str = "outputs",
            save_only_last_epoch: bool = False,
    ):
        """
        初始化一个 PyTorch Lightning 模型
        Args:
            tokenizer : T5/MT5/ByT5 分词器
            pretrained_model : T5/MT5/ByT5 的预训练模型
            output_dir (str, optional): 保存模型检查点的输出目录, 默认为 "outputs"
            save_only_last_epoch (bool, optional): 如果为 True, 则只保存最后一个 epoch, 否则将为每个 epoch 保存模型
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.average_training_loss = None  # 训练时的平均 loss
        self.average_validation_loss = None  # 验证时的平均 loss
        self.save_only_last_epoch = save_only_last_epoch

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return output.loss, output.logits  # loss 的计算为 = CrossEntropyLoss(ignore_index=-100)

    def training_step(self, batch, batch_size):  # 自动打开模型的 train() 模型
        """ 当用训练集训练模型时, 执行该代码 """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(  # 直接调用 forward()  源代码只有 self
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(  # 写入日志
            "train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        return loss

    def validation_step(self, batch, batch_size):  # 自动打开模型的 eval() 模型
        """ 当用验证集测试模型时, 执行该代码 """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        # return {'loss': loss, 'input_ids': input_ids, 'labels': labels}  # 因为在验证集测试结束时什么都不做, 故不返回东西

    def test_step(self, batch, batch_size):  # 自动打开模型的 eval() 模型. 因为我没有放入测试集进来, 所以不会执行该段代码
        """ 当用测试集测试模型时, 执行该代码 """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):  # 在该类初始化时即被调用
        """ 配置优化器(optimizers) """
        return AdamW(self.parameters(), lr=0.0001)

    def training_epoch_end(self, training_step_outputs):
        """ 在每一轮训练结束时保存训练了的分词器和模型 """
        # 注释的代码可用于计算平均 train loss
        # self.average_training_loss = np.round(
        #     # torch.stack(): 把多个 2 维的张量凑成一个3维的张量; 多个 3 维的凑成一个4维的张量...以此类推, 也就是在增加新的维度进行堆叠.
        #     torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
        #     4,  # 小数位数后保留 4 位
        # )
        path = f"{self.output_dir}/simple_T5"
        if self.save_only_last_epoch:
            if self.current_epoch == self.trainer.max_epochs - 1:
                self.tokenizer.save_pretrained(path)  # 保存分词器到路径 path 底下
                self.model.save_pretrained(path)  # 保存模型到路径 path 底下
        else:
            self.tokenizer.save_pretrained(path)
            self.model.save_pretrained(path)

    def validation_epoch_end(self, validation_step_outputs):
        """ 在每一轮验证集测试结束时做点什么呢? """
        pass


class myModel_for_Train:
    """ 自定义的 T5 模型类 """

    def __init__(self) -> None:
        """ 初始化自定义的模型类 """
        self.model = None
        self.T5_Model = None
        self.tokenizer = None
        self.device = None
        self.data_module = None

    def from_pretrained(self, model_name="t5-base") -> None:
        """
        加载预训练 T5 模型进行训练/微调
        参数:
            model_name (str, optional): 确切的模型体系结构名称，"t5-base" 或 "t5-large". 默认为 "t5-base".
        """
        self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(
            f"{model_name}", return_dict=True  # 是否返回一个 ~utils.ModelOutput类 而不是普通的元组
        )

    def train(
            self,
            train_df: pd.DataFrame,
            eval_df: pd.DataFrame,
            source_max_token_len: int = 512,
            target_max_token_len: int = 512,
            batch_size: int = 8,
            max_epochs: int = 5,
            use_gpu: bool = True,
            output_dir: str = "outputs",
            early_stopping_patience_epochs: int = 2,  # 0 表示禁用提前停止功能
            precision=32,
            logger="default",
            dataloader_num_workers: int = 2,
            save_only_last_epoch: bool = False,  # 不设置只保留最后一轮, 而是保留在验证集上效果最好的一轮
    ):
        """
        在自定义数据集上训练 T5 模型
        参数:
            train_df (pd.DataFrame): training dataframe. Dataframe 必须有 2 列 --> "source_text" 和 "target_text"
            eval_df ([type], optional): validation dataframe. Dataframe 必须有 2 列 --> "source_text" 和 "target_text"
            source_max_token_len (int, optional): 源文本的最大 token 长度. 默认值为 512.
            target_max_token_len (int, optional): 目标文本的最大 token 长度. 默认值为 512.
            batch_size (int, optional): batch size. 默认值为 8.
            max_epochs (int, optional): 最大的 epochs 值. 默认为 5.
            use_gpu (bool, optional): 如果为True, 则模型使用 gpu 进行训练. 默认为 True.
            output_dir (str, optional): 保存模型 checkpoints 的输出目录. 默认为 "outputs".
            early_stopping_patience_epochs (int, optional): 在 epoch 结束时监视 val_loss, 如果 val_loss 在指定的 epoch 数之后没有改善(即下降),
            则停止训练. 若设置 0 表示禁用提前停止. 默认为 0(禁用)
            precision (int, optional): 设置精度训练-双精度(64), 全精度(32)或半精度(16). 默认值为 32.
            logger (pytorch_lightning.loggers) : PyTorch Lightning支持的任何记录器. 默认为 "default". 如果为 "default",
            则使用 pytorch lightning default logger, 用于记录训练过程.
            dataloader_num_workers (int, optional): 设置加载 train/test/val dataloader 的进程数量
            save_only_last_epoch (bool, optional): 如果为 True, 则仅保存最后一个 epoch, 否则将保存每个 epoch 的分词器和模型
        """
        self.data_module = myLightningDataModule(
            train_df=train_df,
            test_df=eval_df,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            num_workers=dataloader_num_workers,
        )

        self.T5_Model = myLightningModel(
            tokenizer=self.tokenizer,
            model=self.model,
            output_dir=output_dir,  # 保存 tokenizer 和 model 的路径
            save_only_last_epoch=save_only_last_epoch,  # 只保存最后一轮的 checkpoint
        )

        # 添加回调方法, 用于显示模型训练的进度, 更新频率为 5
        callbacks = [TQDMProgressBar(refresh_rate=5)]

        if early_stopping_patience_epochs > 0:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=early_stopping_patience_epochs,
                verbose=True,
                mode="min",
            )
            callbacks.append(early_stop_callback)

        # 如果有 gpu, 则添加
        gpus = 1 if use_gpu else 0

        # 添加 logger(日志器)
        loggers = True if logger == "default" else logger

        # prepare trainer(训练器)
        trainer = pl.Trainer(
            default_root_dir='./',  # 日志 和 checkpoint 的路径
            logger=loggers,
            enable_checkpointing=False,  # 不保存 checkpoint
            callbacks=callbacks,
            max_epochs=max_epochs,
            gpus=gpus,
            precision=precision,
            log_every_n_steps=1,  # 每训练 1 步(step)就记录一下日志.
        )

        # fit trainer(训练器)
        trainer.fit(self.T5_Model, self.data_module)

    def load_model(
            self,
            model_dir: str = "outputs",
            use_gpu: bool = False
    ):
        """
        加载某一个 checkpoint, 即加载模型
        参数:
            model_type (str, optional): "t5" 或 "mt5". 默认为 "t5".
            model_dir (str, optional): 模型目录的路径. 默认为 "outputs".
            use_gpu (bool, optional): 如果为 True, 模型使用 gpu 进行推理/预测. 默认为 True.
        """

        self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
        self.tokenizer = T5Tokenizer.from_pretrained(f"{model_dir}")

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise "exception ---> no gpu found. set use_gpu=False, to use CPU"
        else:
            self.device = torch.device("cpu")

    def predict(
            self,
            source_text: str,
            max_length: int = 512,
            num_return_sequences: int = 1,
            num_beams: int = 4,  # 按照 t5 论文里写的来
            top_k: int = 50,
            top_p: float = 0.95,
            # do_sample: bool = True,
            repetition_penalty: float = 2.5,
            length_penalty: float = 0.6,  # 按照 t5 论文里写的来
            early_stopping: bool = True,
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = True,
    ):
        """
        生成 T5 模型的预测(文本)
        参数:
            source_text (str): 任何用于生成预测的文本, 即源文本
            max_length (int, optional): 预测的最大 token 长度. 默认值为 512.
            num_return_sequences (int, optional): 要返回的预测数. 默认值为 1.
            num_beams (int, optional): beams 搜索的数量. 默认值为 2.
            top_k (int, optional): 用于 top-k 筛选的最高概率词汇表 tokens 的数量. 默认值为 50.
            top_p (float, optional): 如果设置为 float < 1, 那只有概率加起来等于 top_p 或更高的最可能 token 的最小集被保留用于生成. 默认值为 0.95.
            do_sample (bool, optional): 是否使用抽样; 否则使用贪婪解码.默认值为 True.
            repetition_penalty (float, optional): 重复惩罚的参数. 1.0 意味着没有惩罚. 更多细节请参见[本文]
            (https://arxiv.org/pdf/1909.05858.pdf)。默认值为 2.5.
            length_penalty (float, optional): 基于 beam 生成所使用的长度的指数惩罚. length_penalty > 0.0 表示促进更长的序列,
            而 length_penalty < 0.0 表示鼓励较短的序列. 默认值为 1.0.
            early_stopping (bool, optional): 是否在每批至少有 num_beams 条语句完成时停止 beam search. 默认值为 True.
            skip_special_tokens (bool, optional): 是否跳过特殊的 token, 例如 <pad>, 默认值为 True.
            clean_up_tokenization_spaces (bool, optional): 是否清理 tokens 的空间. 默认值为 True.
        返回:
            list[str]: 返回预测的文本, 即摘要
        """
        input_ids = self.tokenizer.encode(
            source_text,
            return_tensors="pt",
            add_special_tokens=True
        )
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            input_ids = input_ids.to(self.device)  # 放入 gpu
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
        )

        predictions_text = [
            self.tokenizer.decode(
                every_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for every_ids in generated_ids
        ]
        return predictions_text


if __name__ == '__main__':
    # 如果想训练完直接测试, 则下面这个 flag 为 True; 如果想调用训练好的模型, 则设为 False
    train_after_and_test_flag = True

    if train_after_and_test_flag is True:
        torch.cuda.empty_cache()
        pl.seed_everything(42)  # 设置随机种子, 方便复现
        train_data_path = "train.csv"
        t5_train_df = pd.read_csv(train_data_path, sep='\t')
        print('显示前 5 个样例:\n', t5_train_df.head(), '\n总的新闻和摘要对数为::', len(t5_train_df))

        # T5 模型要求数据帧(dataframe)有2列: "source_text" 和 "target_text"
        t5_train_df = t5_train_df.rename(columns={"summary": "target_text", "document": "source_text"})
        t5_train_df = t5_train_df[['source_text', 'target_text']]

        # T5 模型需要一个与任务相关的前缀(prefix): 因为它是一个摘要任务, 我们将添加一个前缀 "summary:"
        t5_train_df['source_text'] = "summarize: " + t5_train_df['source_text']
        print('显示新改造的所有(新闻和摘要对)样例:\n', t5_train_df)  # 平均的文档长度为 1212, 摘要长度为 82

        t5_train_df, t5_valid_df = train_test_split(t5_train_df, test_size=0.1)
        print(t5_train_df.shape, t5_valid_df.shape)

        # ****************  以下是模型的训练代码  ****************
        t5_model = myModel_for_Train()
        t5_model.from_pretrained(model_name="t5-base")
        t5_model.train(
            train_df=t5_train_df,
            eval_df=t5_valid_df,
            source_max_token_len=768,
            target_max_token_len=64,
            batch_size=4,
            max_epochs=10,
            use_gpu=True,
        )
        print('模型已经训练好了！')
    else:
        t5_model, t5_train_df, t5_valid_df = None, None, None

    # ****************  以下是验证集的Rouge评测和生成测试  ****************
    if train_after_and_test_flag is True:
        test_t5_model = t5_model
    else:
        test_t5_model = myModel_for_Train()
        test_t5_model.load_model("outputs/simple_T5", use_gpu=True)
    print('模型已经加载好了！')
    # 对验证集进行 ROUGE 分数计算
    my_rouge = Rouge()
    rouge_1, rouge_2, rouge_l_f1, rouge_l_p, rouge_l_r = 0, 0, 0, 0, 0
    for ind in range(len(t5_valid_df)):
        input_text = t5_valid_df.iloc[ind]['source_text']
        output_text = test_t5_model.predict(input_text)
        label_text = t5_valid_df.iloc[ind]['target_text']
        result = my_rouge.get_scores(output_text, [label_text], avg=True)  # 取一个 batch 的平均
        rouge_1 += result['rouge-1']['f']
        rouge_2 += result['rouge-2']['f']
        rouge_l_f1 += result['rouge-l']['f']
        rouge_l_p += result['rouge-l']['p']
        rouge_l_r += result['rouge-l']['r']
        # print(ind, rouge_1 / (ind + 1), rouge_2 / (ind + 1), rouge_l_f1 / (ind + 1), rouge_l_p / (ind + 1),
        #       rouge_l_r / (ind + 1))
    print('验证集平均的 Rouge_1: {}, Rouge_2: {}, Rouge_l_f1: {}, Rouge_l_p: {}, Rouge_l_r: {}'.format(
        rouge_1 / len(t5_valid_df), rouge_2 / len(t5_valid_df), rouge_l_f1 / len(t5_valid_df),
        rouge_l_p / len(t5_valid_df), rouge_l_r / len(t5_valid_df)))

    text_to_summarize = """summarize: Rahul Gandhi has replied to Goa CM Manohar Parrikar's letter,
        which accused the Congress President of using his "visit to an ailing man for political gains".
        "He's under immense pressure from the PM after our meeting and needs to demonstrate his loyalty by attacking me,"
        Gandhi wrote in his letter. Parrikar had clarified he didn't discuss Rafale deal with Rahul.
        """
    print('生成的摘要为:', test_t5_model.predict(text_to_summarize))


