import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


def load_conversation_data(data_path):
    conversations = []
    labels = []
    lsm_values = []
    m_rlsm_values = []

    # 遍历路径下的所有json文件
    for filename in os.listdir(data_path):
        if filename.endswith('.json'):
            file_path = os.path.join(data_path, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 提取global_metrics中的 LSM 和 M_rLSM
                global_metrics = data.get("global_metrics", {})
                lsm = global_metrics.get('LSM', 0.0)
                m_rlsm = global_metrics.get('M_rLSM', 0.0)
                lsm_values.append(lsm)
                m_rlsm_values.append(m_rlsm)

                # 提取 user_data 中的 label
                user_data = data.get("user_data", {})
                label = user_data.get('label', 0)
                labels.append(label)

                # 提取对话数据 conversations
                conversation_data = []
                for conv in user_data.get("conversations", []):
                    statement = conv.get('statement', '')
                    function_words_percentage = conv.get('function_words_percentage', 0.0)
                    rlsm = conv.get('rLSM', None)  # 处理 rLSM 值，可能为空
                    conversation_data.append({
                        'statement': statement,
                        'function_words_percentage': function_words_percentage,
                        'rLSM': rlsm
                    })

                # 将提取的对话数据加入 conversations 列表
                conversations.append(conversation_data)

    return conversations, labels, lsm_values, m_rlsm_values


class CustomDataset(Dataset):
    def __init__(self, conversations, labels, lsm_values, m_rlsm_values, tokenizer, max_length=40, max_sentences=20):  # max_length与max_sentences的设置需要根据实际情况进行调整,如何确定这两个值？
        # max_length表示每个句子分词后所包含的 token 的最大数量
        self.conversations = conversations
        self.labels = labels
        self.lsm_values = lsm_values
        self.m_rlsm_values = m_rlsm_values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_sentences = max_sentences  # 设置最大句子数

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        try:
            conv = self.conversations[idx]
            label = self.labels[idx]
            lsm = self.lsm_values[idx]
            m_rlsm = self.m_rlsm_values[idx]

            # 截断或填充句子数量
            # 如果对话集中的句子数量超过 max_sentences，则截断；如果少于，则填充。
            if len(conv) > self.max_sentences:
                conv = conv[:self.max_sentences]
            else:
                # 使用空句子进行填充，确保总共的句子数量为 max_sentences
                while len(conv) < self.max_sentences:
                    conv.append({'statement': '', 'function_words_percentage': 0.0, 'rLSM': 0.0})

            # 确保所有语句都是字符串类型，并处理可能的 None 或空值
            statements = [str(c['statement']) if c['statement'] else "" for c in conv]
            function_words_percentages = [c['function_words_percentage'] for c in conv]
            rlsm_values = [c['rLSM'] if c['rLSM'] is not None else 0 for c in conv]  # 处理最后一句没有rLSM的情况

            # 分别对每个语句进行编码，然后手动堆叠结果
            all_input_ids = []
            all_attention_masks = []
            
            for statement in statements:
                inputs = self.tokenizer(
                    statement,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                all_input_ids.append(inputs['input_ids'].squeeze(0))
                all_attention_masks.append(inputs['attention_mask'].squeeze(0))
            
            # 堆叠所有的 tensors
            input_ids = torch.stack(all_input_ids)  # (max_sentences, max_length)
            attention_mask = torch.stack(all_attention_masks)  # (max_sentences, max_length)

            # 将 function_words_percentage 和 rLSM 作为逐句特征
            return {
                'input_ids': input_ids,  # (max_sentences, max_length)
                'attention_mask': attention_mask,  # (max_sentences, max_length)
                'function_words_percentage': torch.tensor(function_words_percentages, dtype=torch.float),  # (max_sentences,)
                'rLSM': torch.tensor(rlsm_values, dtype=torch.float),  # (max_sentences,)
                'LSM': torch.tensor(lsm, dtype=torch.float),  # 单一数值
                'M_rLSM': torch.tensor(m_rlsm, dtype=torch.float),  # 单一数值
                'labels': torch.tensor(label, dtype=torch.long)  # 单一数值
            }
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            print(f"Conversation content: {self.conversations[idx]}")
            raise


class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

        # 确定 hidden_size 和 embed_dim
        hidden_size = self.model.config.hidden_size
        embed_dim = hidden_size + 2

        # 动态选择 num_heads
        num_heads = max(
            [i for i in range(1, embed_dim + 1) if embed_dim % i == 0 and i % 2 == 0 and i <= embed_dim // 2])

        # 使用 TransformerEncoder 替代 Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.aggregation_layer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1  # 使用单层TransformerEncoder，具体使用几层需要根据实际情况进行调整，用多少层效果最好？
        )

        # 分类层保持不变
        self.fc = nn.Linear(embed_dim + 2, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, function_words_percentage, rLSM, LSM, M_rLSM):
        batch_size, max_sentences, max_length = input_ids.shape
        hidden_size = self.model.config.hidden_size

        # 初始化张量存储每个句子的 [CLS] token 特征
        cls_tokens = torch.zeros(batch_size, max_sentences, hidden_size).to(input_ids.device)

        # 循环处理每个句子
        for i in range(max_sentences):
            sentence_input_ids = input_ids[:, i, :]
            sentence_attention_mask = attention_mask[:, i, :]

            with torch.no_grad():
                outputs = self.model(input_ids=sentence_input_ids, attention_mask=sentence_attention_mask)
                cls_tokens[:, i, :] = outputs.last_hidden_state[:, 0, :]

        # 拼接特征
        combined_features = torch.cat((
            cls_tokens,
            function_words_percentage.unsqueeze(2),
            rLSM.unsqueeze(2)
        ), dim=2)

        # 使用 TransformerEncoder 进行特征聚合
        aggregated_features = self.aggregation_layer(combined_features)

        # 取第一个 token 特征
        cls_token_features_for_dialogue = aggregated_features[:, 0, :]

        # 拼接全局特征
        final_features = torch.cat((
            cls_token_features_for_dialogue,
            LSM.unsqueeze(1),
            M_rLSM.unsqueeze(1)
        ), dim=1)

        final_features = self.dropout(final_features)
        logits = self.fc(final_features)

        return logits


data_path = r'D:\毕业论文\对话数据\json文件夹\付费'
conversations, labels, lsm_values, m_rlsm_values = load_conversation_data(data_path)

# 本地模型存放路径
local_model_path = r"D:\Large Models\QwenQwen2.5-1.5B-Instruct"

# 加载本地的Qwen-2模型
model = AutoModel.from_pretrained(local_model_path)

# 加载本地的Qwen-2分词器
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 划分训练集与验证集
train_conversations, val_conversations, train_labels, val_labels, train_lsm_values, val_lsm_values, train_m_rlsm_values, val_m_rlsm_values = train_test_split(
    conversations, labels, lsm_values, m_rlsm_values, test_size=0.2, random_state=42
)

# 创建数据集
train_dataset = CustomDataset(train_conversations, train_labels, train_lsm_values, train_m_rlsm_values, tokenizer)
val_dataset = CustomDataset(val_conversations, val_labels, val_lsm_values, val_m_rlsm_values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # （batch_size，max_sentences, max_length)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)  # （batch_size，max_sentences, max_length)

# 模型、损失函数和优化器
model = CustomModel(model_name=local_model_path, num_labels=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练和评估
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)  # (batch_size, max_sentences, max_length)
        attention_mask = batch['attention_mask'].to(device)  # (batch_size, max_sentences, max_length)
        function_words_percentage = batch['function_words_percentage'].to(device)  # (batch_size, max_sentences)
        rLSM = batch['rLSM'].to(device)  # (batch_size, max_sentences)
        LSM = batch['LSM'].to(device)  # (batch_size,)
        M_rLSM = batch['M_rLSM'].to(device)  # (batch_size,)
        labels = batch['labels'].to(device)  # (batch_size,)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, function_words_percentage, rLSM, LSM, M_rLSM)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
