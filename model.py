import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch import nn

class PTMPredictionModel(nn.Module):
    def __init__(self, configs, num_labels=1):
        """
        Fine-tuning model for post-translational modification prediction.

        Args:
        Args:
            configs: Contains model configurations like
                     - model.model_name (str)
                     - model.hidden_size (int)
                     - model.freeze_backbone (bool, optional)
                     - model.dropout_rate (float, optional)
            num_labels (int): The number of output labels (default: 2 for binary classification).
        """
        super().__init__()

        # 1. Read from configs
        base_model_name = configs.model.model_name
        hidden_size = configs.model.hidden_size
        freeze_backbone = configs.model.freeze_backbone
        freeze_embeddings = configs.model.freeze_embeddings
        freeze_backbone_layers = configs.model.freeze_backbone_layers
        classifier_dropout_rate = configs.model.classifier_dropout_rate
        backbone_dropout_rate = configs.model.backbone_dropout_rate
        self.use_decoder_block = getattr(configs.model, "use_decoder_block", False)
        esm_to_decoder_dropout_rate = configs.model.last_state_dropout_rate

        # 2. Load the pretrained transformer
        config = AutoConfig.from_pretrained(base_model_name)
        config.torch_dtype = "bfloat16"
        config.classifier_dropout = classifier_dropout_rate

        # print(f"Config: {config}")

        self.base_model = AutoModel.from_pretrained(base_model_name, config=config)

        # 3. Freeze backbone if requested
        if freeze_backbone:
            if freeze_embeddings:
                # Freeze all layers (including embeddings)
                for param in self.base_model.parameters():
                    param.requires_grad = False
                # Unfreeze layers
                for i, layer in enumerate(self.base_model.encoder.layer):
                    if i >= freeze_backbone_layers:
                        for param in layer.parameters():
                            param.requires_grad = True
                        # add dropout to unfrozen backbone layers
                        layer.attention.self.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.attention.output.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.intermediate.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.output.dropout = nn.Dropout(backbone_dropout_rate)
                        # print("Unfreezing layer", i+i)
            else:
                # Freeze requested layers and leave embeddings
                for i, layer in enumerate(self.base_model.encoder.layer):
                    if i < freeze_backbone_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        layer.attention.self.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.attention.output.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.intermediate.dropout = nn.Dropout(backbone_dropout_rate)
                        layer.output.dropout = nn.Dropout(backbone_dropout_rate)

        # 4. Add condition embeddings and decoder block
        if self.use_decoder_block:
            # Condition embedding
            self.condition_embedding = nn.Embedding(
                num_embeddings=configs.model.num_conditions,
                embedding_dim=hidden_size
            )

            self.encoder_to_decoder_dropout = nn.Dropout(esm_to_decoder_dropout_rate)

            # Transformer decoder with cross-attention
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=configs.transformer_head.num_heads,
                dim_feedforward=configs.transformer_head.dim_feedforward,
                dropout=configs.transformer_head.dropout,
                batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=configs.transformer_head.num_layers
            )

            self.norm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(classifier_dropout_rate)

        # 5. Add classifier on top
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, condition_idx=None):
        """
        Forward pass for the PTM prediction model.

        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention mask for padding.
            condition_idx (Tensor, optional): Condition indices for the decoder block.

        Returns:
            Tensor: Logits for each residue in the input sequence.
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # Shape: [batch_size, seq_len, hidden_size]
        if self.use_decoder_block:
            sequence_output = self.encoder_to_decoder_dropout(sequence_output)
            condition_repr = self.condition_embedding(condition_idx).unsqueeze(1)
            sequence_output = self.transformer_decoder(
                tgt=sequence_output,
                memory=condition_repr
            )
            sequence_output = self.norm(sequence_output)
            sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        return logits

    def num_parameters(self):
        """
        Returns the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def prepare_model(configs):
    """
    Prepares the ESM2 model and tokenizer based on given configurations.

    Args:
        configs (dict): A dictionary containing model configurations.
            Example keys:
                - "model_name" (str): The name of the ESM model to load (e.g., "facebook/esm2_t12_35M_UR50D").
                - "hidden_size" (int): The hidden size of the model
                - maybe more to be added ?

    Returns:
        tokenizer: The tokenizer for the ESM2 model.
        model: The ESM2 model loaded with the specified configurations.
    """
    # Extract configurations
    model_name = configs.model.model_name

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    model = PTMPredictionModel(configs)

    print(f"Loaded model: {model_name}")
    print(f"Model has {model.num_parameters():,} trainable parameters")

    return tokenizer, model

if __name__ == '__main__':
    # This is the main function to test the model's components
    print("Testing model components")

    from box import Box
    import yaml

    config_file_path = 'configs/config.yaml'
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    test_configs = Box(config_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = prepare_model(test_configs)
    model.to(device)

    # Define a sample protein sequence
    # randomly generated protein sequence, not really sure if it's valid
    protein_sequence = "MVLSPADKTNVKAAWGKVGAHAGEY"
    labels = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]], dtype=torch.long)


    # Tokenize the input sequence
    inputs = tokenizer(protein_sequence, return_tensors="pt", padding=True, truncation=True, max_length=64, add_special_tokens=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    labels = labels.to(device)

    # Forward pass through the model
    with torch.no_grad():  # Disable gradient computation for inference
        condition_idx = torch.tensor([0], device=device)
        logits = model(**inputs, condition_idx=condition_idx)
        print(f"Logits shape: {logits.shape}")  # Shape: [batch_size, sequence_length, num_labels]
        # Print the logits tensor
        # print("Logits values:")
        # print(logits)

    # print(model)