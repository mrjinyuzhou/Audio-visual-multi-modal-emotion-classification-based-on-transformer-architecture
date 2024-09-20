import torch
import torchvision.models as models
import torch.nn as nn
import torch.onnx
from torchsummary import summary
from torchvision.models import resnet50,resnet18,efficientnet_b0,efficientnet_b1,resnet152, vit_b_16, ViT_B_16_Weights

class VIT(nn.Module):
    def __init__(self, phase='train'):
        super(VIT, self).__init__()
        
        if phase == 'train':
            vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            vit = vit_b_16()  # 不加载预训练权重
        
        self.feature_extractor = vit
        self.fc_emotion = nn.Sequential(
            nn.Linear(1000, 512),
            nn.GELU(),  # 使用 GELU 代替 ReLU
            nn.Linear(512, 256)
        )
        
        # 替换 ViT 模型中所有的 ReLU 为 GELU (可选)
        # for module in self.modules():
        #     if isinstance(module, nn.ReLU):
        #         new_module = nn.GELU()
        #         setattr(module, 'new_module', new_module)
        
    def forward(self, x):
        # x = x.permute(0, 2, 3, 1)
        features = self.feature_extractor(x)
        # print('features', features.shape)
        vit_logits = self.fc_emotion(features)
        # print('vit_logits', vit_logits.shape)

        return vit_logits
    
class Transformer(nn.Module):
    def __init__(self, input_size=1, num_heads=8, hidden_dim=512, num_layers=6, output_size=512, sequence_length=200, phase='train'):
        super(Transformer, self).__init__()
        
        self.input_embedding = nn.Linear(input_size, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*4, 
            activation='gelu', 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, output_size)
        )
        
        if phase == 'train':
            self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_embedding.weight)
        nn.init.xavier_uniform_(self.fc[0].weight)
        nn.init.xavier_uniform_(self.fc[2].weight)

    def forward(self, x):  # x: [batch_size, sequence_length]
        seq_length = x.size(1)  # 获取当前序列长度

        # Ensure positional_encoding fits current sequence length
        if seq_length > self.positional_encoding.size(1):
            raise ValueError(f"Input sequence length ({seq_length}) exceeds positional encoding length ({self.positional_encoding.size(1)}).")
        
        # Truncate positional_encoding without modifying self.positional_encoding
        positional_encoding = self.positional_encoding[:, :seq_length, :]
        
        x = x.unsqueeze(-1)  # [batch_size, sequence_length, 1]
        embedded_input = self.input_embedding(x)  # [batch_size, sequence_length, hidden_dim]
        embedded_input = embedded_input + positional_encoding  # [batch_size, sequence_length, hidden_dim]
        transformer_output = self.transformer_encoder(embedded_input)  # [batch_size, sequence_length, hidden_dim]
        # transformer_features = transformer_output[:, -1, :]  # [batch_size, hidden_dim]
        output = self.fc(transformer_output)  # [batch_size, sequence_length, output_size]
        return output


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Define multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        # Define a feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, query_features, key_value_features):
        """
        query_features: (batch_size, sequence_length, hidden_dim)  # Image features
        key_value_features: (batch_size, sequence_length, hidden_dim)  # Audio features
        """
        
        # Transpose to fit MultiheadAttention input
        query_features = query_features.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        key_value_features = key_value_features.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        # Apply multi-head attention
        attn_output, _ = self.attention(query=query_features, key=key_value_features, value=key_value_features)
        
        # Reshape back to original
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
        
        # Apply feedforward layer
        output = self.feedforward(attn_output)
        
        # Add & Norm
        output = self.layer_norm1(attn_output + output)
        output = self.layer_norm2(output + self.feedforward(output))
        
        return output

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Define multi-head attention with batch_first=True
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Define a feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, query_features, key_value_features):
        """
        query_features: (batch_size, query_len, hidden_dim)  # Image features
        key_value_features: (batch_size, key_len, hidden_dim)  # Audio features
        """
        
        # Apply multi-head attention
        attn_output, _ = self.attention(query=query_features, key=key_value_features, value=key_value_features)
        
        # Apply feedforward layer
        output = self.feedforward(attn_output)
        
        # Add & Norm
        output = self.layer_norm1(attn_output + output)
        output = self.layer_norm2(output + self.feedforward(output))
        
        return output


class MultimodalModel(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=256, num_transformer_layers=3):
        super(MultimodalModel, self).__init__()
        self.image_extractor = VIT(phase='train')  # 确保 VIT 输出为 [batch_size, hidden_dim]
        self.audio_extractor = Transformer(
            input_size=1, 
            num_heads=8, 
            hidden_dim=hidden_dim, 
            num_layers=6, 
            output_size=hidden_dim, 
            sequence_length=200,  # 根据实际序列长度设置
            phase='train'
        )
        
        # Cross-Attention Fusion Module
        self.fusion_module = CrossAttentionLayer(hidden_dim=hidden_dim, num_heads=8)
        
        # Define stacked Transformer layers after cross-attention
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                activation='gelu',
                batch_first=True  # 确保 batch_first=True
            ) for _ in range(num_transformer_layers)
        ])
        
        # Final output layer (after transformer layers), for 3-class classification
        self.fc_output = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)  # 三分类任务的输出层
        )

    def forward(self, audio_input, image_input):
        # Extract audio and image features
        image_features = self.image_extractor(image_input)  # [batch_size, hidden_dim]
        # print('image_features___' , image_features.shape)
        audio_features = self.audio_extractor(audio_input)  # [batch_size, sequence_length, hidden_dim]
        # print('audio_features___' , audio_features.shape)
        
        # Reshape image_features to [batch_size, 1, hidden_dim] for cross-attention
        image_features = image_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Fuse audio and image features with cross-attention
        fused_features = self.fusion_module(audio_features, image_features)  # [batch_size, hidden_dim]
        # print('fused_features', fused_features.shape)
        
        # Add a sequence dimension for transformer layers
        # fused_features = fused_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Pass through stacked Transformer layers
        for transformer_layer in self.transformer_layers:
            fused_features = transformer_layer(fused_features)  # [batch_size, 1, hidden_dim]
        
        # Pooling the output (e.g., mean pooling) after transformer layers
        pooled_features = fused_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Final output for 3-class classification
        output = self.fc_output(pooled_features)  # [batch_size, 3]
        
        return output

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
if __name__ == "__main__":
    # 参数设置
    batch_size = 8
    sequence_length = 200
    input_size_audio = 1
    input_size_image = 3  # 对于图像，假设每个通道作为一个特征
    hidden_dim = 512
    num_classes = 3

    # 构造假音频数据 (batch_size, sequence_length, input_size)
    audio_data = torch.randn(batch_size, sequence_length)

    # 构造假图像数据 (batch_size, 3, 224, 224)，模拟 (batch_size, channels, height, width) 的输入
    # 这里假设输入图像为 224x224 大小的 3 通道图像
    image_data = torch.randn(batch_size, 3, 224, 224)

    # 构造假标签 (batch_size)，三分类任务的标签
    labels = torch.randint(0, num_classes, (batch_size,))
    print(audio_data.shape, image_data.shape, labels.shape)
    model = MultimodalModel(num_classes=3)
    
    output = model(audio_data, image_data)
    
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")
    
    print(output.shape)