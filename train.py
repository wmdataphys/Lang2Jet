import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from torch.utils.data import DataLoader
import vector
from omegaconf import OmegaConf
import importlib
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from transformers.models.distilbert import DistilBertForSequenceClassification as MODEL
from transformers.models.bert.modeling_bert import BertForSequenceClassification as BERT_MODEL

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse
from datetime import datetime
import time
vector.register_awkward()
import psutil
from sklearn.metrics import roc_auc_score

from data.dataset import MultiClassJetDataset
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger.info("Setup complete")


def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024  # in MB
    print(f"[MEMORY] {tag} → {mem_mb:.2f} MB")
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass
    
def count_items_in_loader(data_loader):
    count = 0
    for batch in data_loader:
        if isinstance(batch["jet_type_labels"], torch.Tensor):
            count += batch["jet_type_labels"].size(0)  # batch size
        else:
            count += 1  # fallback if single items
    return count

def save_checkpoint(model, optimizer, epoch, best_val_loss, path, training_metric, validation_metric, lr_history, counter):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'training_metric': training_metric,
        'validation_metric': validation_metric,
        'lr': lr_history,
        'counter': counter

    }, path)

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    training_metric = checkpoint.get('training_metric', {"loss": [], "accuracy": []})
    validation_metric = checkpoint.get('validation_metric', {"loss": [], "accuracy": []})
    lr_history = checkpoint['lr']
    counter = checkpoint.get('counter', 0)
    return epoch, best_val_loss, training_metric, validation_metric, lr_history, counter

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layer, output_dim,device):
        super(MLP, self).__init__()
        self.device=device
        # Build layers dynamically with Sequential
        layer_dims = [input_dim] + hidden_layer + [output_dim]

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:  # no activation after last layer
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):    
        return self.mlp(x)
    
class BERT_MLP(BERT_MODEL, nn.Module):
    def __init__(self, config, mlp_model, mdl = None, fix_backbone = False, layer_list = None, **kwargs):
        super().__init__(config)
       
        self.relu = nn.ReLU()
        self.bert.embeddings = mlp_model
        if mdl is not None:
            for (c_name, c_parameter) in self.named_parameters():
                try:
                    b_parameter = mdl.get_parameter(c_name).data
                    if (b_parameter.shape == c_parameter.shape):
                        c_parameter.data = mdl.get_parameter(c_name).data
                except:
                    print(f"Parameter {c_name} not found in the model")
                    pass


        if fix_backbone and mdl is not None:
            for child_name, child_module in self.named_children():
                for sub_child_name, sub_child_module in child_module.named_children():
                    if sub_child_name == 'encoder' or sub_child_name == 'pooler':
                        for param_name, param in sub_child_module.named_parameters():
                            if "embeddings" not in param_name:
                                param.requires_grad = False
                            
    def forward(self, x, attention_mask):

        x_mlp = self.bert.embeddings(x)         # pass through the MLP model
        x = x_mlp * attention_mask.unsqueeze(-1)
        attention_mask = attention_mask.to(dtype=torch.float)
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0  # convert to additive mask

        # pass through the bert model
        head_mask = self.bert.get_head_mask(None, self.config.num_hidden_layers)
        bert_output = self.bert.encoder(x, attention_mask=extended_mask, head_mask=head_mask)
        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        pooled_output = self.bert.pooler(hidden_state)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        return logits
    
    
    
class CustomBERT(MODEL, nn.Module):
    def __init__(self, config, mlp_model,
                 mdl = None, fix_backbone = False,
                use_preclassifier=False, bias = False,
                use_layer_norm=False, mean_pooling=False, attention_pooling=False, attend_only_nonpadded_values=False, use_lightweight=True, **kwargs):
        super().__init__(config)
      
        self.relu = nn.ReLU()
        self.mean_pooling = mean_pooling
        self.attention_pooling = attention_pooling
        self.attention = nn.Linear(config.dim, 1)
        self.attend_only_nonpadded_values = attend_only_nonpadded_values
        
        custom_head = nn.Sequential(
            nn.Linear(config.dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_labels)
        )
        custom_head_lightweight = nn.Sequential(
                nn.Linear(config.dim, 10),
                nn.ReLU(),
                nn.Linear(10, config.num_labels)
        )
        if use_lightweight:
            print("Using lightweight custom head with 10 hidden units")
            self.classifier = custom_head_lightweight
        else:
            self.classifier = custom_head
        self.distilbert.embeddings = mlp_model
        del self.pre_classifier  # Remove pre_classifier if it exists
        if mdl is not None:
            for (c_name, c_parameter) in self.named_parameters():
                try:
                    b_parameter = mdl.get_parameter(c_name).data
                    if (b_parameter.shape == c_parameter.shape):
                        c_parameter.data = mdl.get_parameter(c_name).data
                except:
                    print(f"Parameter {c_name} not found in the model")
                    pass
        if fix_backbone and mdl is not None:
            for child_name, child_module in self.named_children():
                if child_name == 'distilbert': #or child_name == 'pre_classifier':
                    for param_name, param in child_module.named_parameters():
                        # freeze everything first
                        param.requires_grad = False 
                        if ("embeddings" in param_name
                            or ("bias" in param_name and bias)
                            # or (use_preclassifier and child_name == "pre_classifier")
                            or ("layer_norm" in param_name and use_layer_norm)
                            ):
                            param.requires_grad = True
                            
    def forward(self, x, attention_mask):
        # pass through the MLP model
        x_mlp = self.distilbert.embeddings(x)
        x = x_mlp * attention_mask.unsqueeze(-1)
        # attention_mask = attention_mask.to(dtype=torch.bool)
        attention_mask = attention_mask.to(dtype=torch.float)
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0  # convert to additive mask

        # pass through the distilbert model
        head_mask = self.distilbert.get_head_mask(None, self.config.num_hidden_layers)
        distilbert_output = self.distilbert.transformer(x, attn_mask=extended_mask, head_mask=head_mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        if self.mean_pooling and not self.attention_pooling:
            pooled_output = hidden_state.mean(dim=1)
        elif self.attention_pooling and not self.mean_pooling and not self.attend_only_nonpadded_values:
            weights = torch.softmax(self.attention(hidden_state), dim=1)
            pooled_output = (weights * hidden_state).sum(dim=1)
        elif self.attention_pooling and not self.mean_pooling and self.attend_only_nonpadded_values:
            scores = self.attention(hidden_state)   # (bs, seq_len, 1)
            scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)  # mask padded positions before softmax
            weights = torch.softmax(scores, dim=1)
            pooled_output = (weights * hidden_state).sum(dim=1)
        else:
            pooled_output = hidden_state[:, 0]  # (bs, dim)
        # pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        # pooled_output = self.relu(pooled_output)  # (bs, dim)
        # pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        return logits

class CustomBERT_customclassifier(MODEL, nn.Module):
    def __init__(self, config, mlp_model,
                 mdl = None, fix_backbone = False,
                 bias = False,
                 **kwargs):
        super().__init__(config)
      
        self.relu = nn.ReLU()
        custom_head = nn. Sequential(
            nn.Linearconfig.dim, 256)
        # custom_head = nn.Sequential(
        #     nn.Linear(config.dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, config.num_labels)
        # )

        del self.pre_classifier  # Remove pre_classifier if it exists
        self.classifier = custom_head
        self.distilbert.embeddings = mlp_model
        if mdl is not None:
            for (c_name, c_parameter) in self.named_parameters():
                try:
                    b_parameter = mdl.get_parameter(c_name).data
                    if (b_parameter.shape == c_parameter.shape):
                        c_parameter.data = mdl.get_parameter(c_name).data
                except:
                    print(f"Parameter {c_name} not found in the model")
                    pass
        if fix_backbone and mdl is not None:
            for child_name, child_module in self.named_children():
                if child_name == 'distilbert':
                    for param_name, param in child_module.named_parameters():
                        # freeze everything first
                        param.requires_grad = False 
                        if ("embeddings" in param_name
                            or ("bias" in param_name and bias)
                            ):
                            param.requires_grad = True
                            
    def forward(self, x, attention_mask):
        # pass through the MLP model
        x_mlp = self.distilbert.embeddings(x)
        x = x_mlp * attention_mask.unsqueeze(-1)
        # attention_mask = attention_mask.to(dtype=torch.bool)
        attention_mask = attention_mask.to(dtype=torch.float)
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0  # convert to additive mask

        # pass through the distilbert model
        head_mask = self.distilbert.get_head_mask(None, self.config.num_hidden_layers)
        distilbert_output = self.distilbert.transformer(x, attn_mask=extended_mask, head_mask=head_mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = (hidden_state* attention_mask.unsqueeze(-1)).sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(-1) # (bs, dim)
        # pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        return logits


    
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler=None, debug=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    total_batches = 0
    backprop_times = []
    start_total_time = time.perf_counter()

    all_train_logits = []
    all_train_preds = []
    all_train_labels = []

    # Debug: Check if dropout is active in training
    print("\n=== TRAINING PHASE ===")
    lr_history = []
    max_batch_train = len(train_loader)
    if debug:
        max_batch_train = 5
        print (f"\n ========= TRAINING IN DEBUG MODE: MAX_BACTCH SET TO {max_batch_train} \n")

    else:
        print(f"\n ========= TRAINING IN NORMAL MODE: BEGINNING TRAINING \n")
    
    # print("MLP weight before training:", before_weight)
    for _, batch in zip(range(max_batch_train), tqdm(train_loader, desc="Training")):
        if 'part_features' in batch:
            inputs = batch['part_features']
            att_mask = batch['part_mask']
            labels = batch['jet_type_labels']
        else:
            inputs = batch[0]
            att_mask = batch[1]
            labels = batch[2]
            

        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=att_mask)
        loss = criterion(outputs, labels)
        logits = outputs

        start_time = time.perf_counter()
        loss.backward()
        backprop_times.append(time.perf_counter() - start_time)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_train_logits.append(logits.detach().cpu())
        all_train_preds.append(predicted.detach().cpu())
        all_train_labels.append(labels.detach().cpu())

        total_batches += 1

    total_train_time = time.perf_counter() - start_total_time
    avg_backprop_time = sum(backprop_times) / len(backprop_times)

    # after_weight = model.mlp_model.fc1.weight[0, 0].item()
    # print("MLP weight after training:", after_weight)

    print(f"\n[Train] Average backprop time/batch: {avg_backprop_time:.4f} sec")
    print(f"[Train] Total training time: {total_train_time:.2f} sec")
    print(f"[Train] Train accuracy: {correct}/{total} = {correct/total:.4f}")
    print(f"[Train] Predicted class distribution: {np.bincount(torch.cat(all_train_preds).numpy())}")
    print(f"[Train] True label distribution:      {np.bincount(torch.cat(all_train_labels).numpy())}")

    train_loss = total_loss / total_batches
    train_acc = correct / total

    # Validation loop
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    all_val_logits = []
    all_val_preds = []
    all_val_labels = []
    total_val_batches = 0
    max_batch_val = len(val_loader)
    if debug:
        max_batch_val = 5
        print (f"\n ========= TRAINING IN DEBUG MODE: MAX_BACTCH SET TO {max_batch_val} \n")

    else:
        print(f"\n ========= TRAINING IN NORMAL MODE: BEGINNING VALIDATING \n")
    with torch.no_grad():
        for _ , batch in zip(range(max_batch_val), tqdm(val_loader, desc="Validation")):
            if 'part_features' in batch:
                inputs = batch['part_features']
                att_mask = batch['part_mask']
                labels = batch['jet_type_labels']
            else:
                inputs = batch[0]
                att_mask = batch[1]
                labels = batch[2]
            outputs = model(inputs, attention_mask=att_mask)
            loss = criterion(outputs, labels)
            logits = outputs

            _, predicted = torch.max(logits, 1)

            val_loss += loss.item()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            all_val_logits.append(logits.cpu())
            all_val_preds.append(predicted.cpu())
            all_val_labels.append(labels.cpu())

            total_val_batches += 1

    val_loss /= total_val_batches
    val_acc = val_correct / val_total
    if scheduler:
        scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    print("current learning rate", current_lr)
    print(f"\n[Val] Accuracy: {val_correct}/{val_total} = {val_acc:.4f}")
    print(f"[Val] Predicted class distribution: {np.bincount(torch.cat(all_val_preds).numpy())}")
    print(f"[Val] True label distribution:      {np.bincount(torch.cat(all_val_labels).numpy())}")

    # Convert to NumPy

    return_dict = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr_history": lr_history
    }
    return return_dict

      
# Lets plot the training and validation curves

def plot_training_curves(training_metric, validation_metric, n_train, cfg):
    save_dir = cfg.get('results')
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('/sciclone/home/hnayak/scr10/Transfer_Learning/style.mplstyle')
    plt.figure()
    plt.plot(training_metric['loss'], label='Training Loss')
    plt.plot(validation_metric['loss'], label='Validation Loss') 
    # plt.title(f"Training and Validation Loss with {n_train:.0f} samples")
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    ax=plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)
    if cfg["save_plots"]:
        loss_path = os.path.join(save_dir, f"loss_curve_{n_train}_samples.pdf")
        plt.savefig(loss_path)
        print(f"Saved loss plot to: {loss_path}")
    plt.close()

    plt.figure()
    plt.plot(training_metric['accuracy'], label='Training Accuracy')
    plt.plot(validation_metric['accuracy'], label='Validation Accuracy')
    # plt.title(f"Training and Validation Accuracy with {n_train:.0f} samples") 
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) 
    plt.legend()
    plt.grid(True)
    if cfg["save_plots"]:
        acc_path = os.path.join(save_dir, f"accuracy_curve_{n_train}_samples.pdf")
        plt.savefig(acc_path)
        print(f"Saved accuracy plot to: {acc_path}")
    plt.show()

# Lets test the model on the test set
def test(model, dataloader, epoch_number, criterion, cfg, n_train, best_model = False, debug=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_batch = 0
    all_preds = []
    all_labels = []
    all_probs = []
    misclassified = []
    max_batch = len(dataloader)
    if debug:
        max_batch = 5
        print (f"\n ========== DEBUG MODE: MAX_BATCH SET TO {max_batch} ========== \n")
    else:
        print ("\n ========== TEST MODE: BEGINNING TEST ========== \n")
        
    with torch.no_grad():
        for _, batch in zip(range(max_batch), tqdm(dataloader, desc="Inference")):
            if 'part_features' in batch:
                inputs = batch['part_features']
                att_mask = batch['part_mask']
                labels = batch['jet_type_labels']
                ids = batch['event_id']
            else:
                inputs = batch[0]
                att_mask = batch[1]
                labels = batch[2]
                ids = batch[3]
            # labels = torch.nn.functional.one_hot(batch['jet_type_labels'], num_classes = 2).to(device)
            outputs = model(inputs, attention_mask = att_mask)
            loss = criterion(outputs, labels)
            logits = outputs
            probs = torch.softmax(logits, dim=1)[:, 1] # [batch_size]
            total_loss += loss.item()
            total_batch += 1
            _, predicted = torch.max(logits, 1)
            mismatches = predicted != labels
            for id_, true, pred, mismatch in zip(ids, labels, predicted, mismatches):
                if mismatch:
                    misclassified.append((id_.item(), true.item(), pred.item()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    auc = roc_auc_score(all_labels, all_probs)
    print(f"AUC: {auc:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["QCD", "t->bqq'"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    report = classification_report(all_labels, all_preds, target_names=["QCD", "t->bqq'"])
    print("Classification Report:\n", report)
    if cfg["save_cm"]:
        test_dir = os.path.join(cfg.get('results'), 'test_results')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir, exist_ok=True)
        cfg['test_result_dir'] = test_dir
        if best_model:
            save_path=f"{cfg['test_result_dir']}/cm_{n_train}samples_{epoch_number}epoch_bestmodel.pdf"
        else:
            save_path=f"{cfg['test_result_dir']}/cm_{n_train}samples_{epoch_number}epoch.pdf"
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved confusion matrix to: {save_path}")
        misclassified_array = np.array(misclassified, dtype=np.int32)  # shape: (num_misclassified, 3)
        mis_file = os.path.join(cfg['test_result_dir'], f"misclassified_epoch_{epoch_number}.npy")
        np.save(mis_file, misclassified_array)
        print(f"Saved misclassified samples to: {mis_file}")
        
    test_metrics = {
        "epoch": epoch_number,
        "loss": total_loss / total_batch,
        "accuracy": correct / total,
        "auc": auc,
        "confusion_matrix": cm,
        "classification_report": report
    }
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    return test_metrics

def run_inference(model, cfg, device, only_test = True, criterion = None, debug = False):

    if not only_test or not criterion:
        print ("Currently inference is not supported")
        return -1
    
    print("loading test dataset")
    test_dataset = MultiClassJetDataset(
        h5FilePath=cfg['data']['test_dir'],
        n_load=cfg['data']['n_load_test']
    )
    test_dataset.to(device)
    print_memory_usage("after loading test")
    n_test = len(test_dataset)
    
    print(f"Test: {n_test} samples")
    test_loader = DataLoader(test_dataset, batch_size=cfg['test_batch_size'], shuffle=False)
    print(f"Loading model from {cfg['best_model_path']}")
    
    model.load_state_dict(torch.load(os.path.join(cfg['best_model_path'])))
    if model.device.type != test_dataset.device():
        model.to(test_dataset.device())
    metrics = test(model, test_loader, 'test', criterion, cfg, n_train=n_test, best_model=True, debug=debug)
    
    data_dict = {}
    for k, v in metrics.items():
        if k != "confusion_matrix":
            data_dict[k] = [v]
    
    # {"test loss" : metrics["loss"], "test accuracy" : metrics["accuracy"], "AUC" : metrics["auc"], "classification report": metrics["classification_report"]}
    
    # for k, v in metrics["auc_dict"].items():
    #     data_dict[f"auc_{cfg['distilBERTConfig']['id2label'][k]}"] = v

    pd.DataFrame(data_dict).to_csv(os.path.join(cfg['test_result_dir'], 'test_results.csv'), index=False)
    
    print (f"Results of the test is {metrics}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(cfg):
   
    print("config:", cfg)
    init_time = time.perf_counter()
    if cfg.get("seed") is not None:
        print("seed:", cfg.get("seed"))
        set_seed(cfg.get("seed"))
        
    if not os.path.exists(cfg.get("save_dir")):
        os.makedirs(cfg.get("save_dir"), exist_ok=True)    
    cfg["results"] = os.path.join(cfg['save_dir'], "results")
    if not os.path.exists(cfg["results"]):
        os.makedirs(cfg["results"], exist_ok=True)
        
    IS_DEBUG = cfg.get("debug", False)
    t=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
    use_preclassifier = cfg.get('use_preclassifier', False)
    bias = cfg.get('update_bias', False)
    layer_norm = cfg.get('use_layer_norm', False)
    mean_pooling = cfg.get('mean_pooling', False)
    attn_pooling = cfg.get('attention_pooling', False)
    attend_only_nonpadded_values = cfg.get('attend_only_nonpadded_values', False)
    use_lightweight = cfg.get('use_lightweight', True)
    print("mean pooling:", mean_pooling)
    print("using preclassifier:", use_preclassifier)
    print("updating bias:", bias)
    print("using layer norm:", layer_norm)
    print("using attention pooling:", attn_pooling)
    print("attend only nonpadded values:", attend_only_nonpadded_values)
    print("using lightweight custom head:", use_lightweight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fix_backbone = cfg["fix_backbone"]
    if cfg["model_type"] == "fix_backbone":
        _config = {"id2label": {"0" : "q/g", "1" : "t->bqq"},
                "label2id": {"q/g" : 0, "t->bqq" : 1},
                "max_position_embeddings": 128
            }
        print(f"Using {cfg['model_type']} model")
        default_distilBERT= AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        config = default_distilBERT.config
        config.update(_config)
        mlp_model = MLP(cfg["input_dim"], cfg["hidden_layer"], cfg["output_dim"], device)
        customBERT = CustomBERT(config,mlp_model, mdl = default_distilBERT, fix_backbone = fix_backbone, use_preclassifier=use_preclassifier, bias= bias, use_layer_norm=layer_norm, mean_pooling=mean_pooling, attention_pooling=attn_pooling, attend_only_nonpadded_values=attend_only_nonpadded_values, use_lightweight=use_lightweight)
    
    elif cfg["model_type"] == "custom_classifier":
        _config = {"id2label": {"0" : "q/g", "1" : "t->bqq"},
                "label2id": {"q/g" : 0, "t->bqq" : 1},
                "max_position_embeddings": 128
            }
        print(f"Using {cfg['model_type']} model")
        default_distilBERT= AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        config = default_distilBERT.config
        config.update(_config)
        mlp_model = MLP(cfg["input_dim"], cfg["hidden_layer"], cfg["output_dim"], device)
        customBERT = CustomBERT_customclassifier(config,mlp_model, mdl = default_distilBERT, fix_backbone = fix_backbone, bias= bias)

    elif cfg["model_type"] == "bert":
        print(f"Using {cfg['model_type']} model")
        _config = {"max_position_embeddings": 128}
        default_bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        config = default_bert.config
        config.update(_config)
        mlp_model = MLP(cfg["input_dim"], cfg["hidden_layer"], cfg["output_dim"], device)
        customBERT = BERT_MLP(config,mlp_model, mdl = default_bert, fix_backbone = fix_backbone)
    
    else:
        raise ValueError(f"Unknown model type: {cfg['model_type']}. Choose from 'fix_backbone', 'full_finetune', or 'train_from_scratch' or 'custom_classifier'.")
    
    print(f"Number of trainable parameters: {count_parameters(customBERT):,}")
    print(customBERT)
    customBERT.to(device)
    for name, param in customBERT.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
    lr = cfg['lr']
    print(f"using learning rate {lr}")
    if cfg["optimizer"] == "adam":
        optimizer = optim.Adam(customBERT.parameters(), lr=lr)
    elif cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(customBERT.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    use_scheduler = cfg.get("use_scheduler", False)
    print("use_scheduler:", use_scheduler)
    scheduler = None
    if use_scheduler:
        print("Using ReduceLROnPlateau scheduler")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=cfg['factor'], patience=cfg['patience']
        )
    num_epochs = cfg["epochs"]
    
    if cfg.get("onlytest", False):
        print("Only testing mode is enabled. Skipping training and validation.")
        run_inference(customBERT, cfg, device, only_test=True, criterion=criterion, debug = IS_DEBUG)
        end_time = time.perf_counter()
        print(f"total time taken: {end_time - init_time:.2f} seconds")
        return 0

    print("loading training dataset")
    print_memory_usage("before loading train")

    print("Using JetClass data")
    train_dataset = MultiClassJetDataset(
    h5FilePath= cfg['data']['train_dir'],
    n_load=cfg['data']['n_load_train'])
    print_memory_usage("after loading train")
    
    val_dataset = MultiClassJetDataset(
    h5FilePath= cfg['data']['val_dir'],
    n_load=cfg['data']['n_load_val'])
    print_memory_usage("after loading val")
    
    print("loading test dataset")
    test_dataset = MultiClassJetDataset(
        h5FilePath= cfg['data']['test_dir'],
        n_load=cfg['data']['n_load_test'])
    print_memory_usage("after loading test")

    train_dataset.to(device)
    val_dataset.to(device)
    test_dataset.to(device)
    print(f"train_dataset pf features shape: {train_dataset.pf_features.shape}, masks shape: {train_dataset.pf_mask.shape}, labels shape: {train_dataset.labels.shape}")
    print("train dataset is in", train_dataset.device())
    print("val dataset is in", val_dataset.device())
    print("test dataset is in", test_dataset.device())
        
    print("DONE")
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_test = len(test_dataset)

    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    total = n_train + n_val + n_test

    print(f"Train ratio: {n_train / total:.2f}")
    print(f"Val ratio:   {n_val / total:.2f}")
    print(f"Test ratio:  {n_test / total:.2f}")

    train_loader = DataLoader(train_dataset, batch_size=cfg['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg['test_batch_size'], shuffle=False)
    print_memory_usage("after moving all to GPU")      
    # Train the model
    training_metric = {"loss" : [], "accuracy" : []}
    validation_metric = {"loss" : [], "accuracy" : []}
    # ckpt_path=f"Bestmodel_{cfg['model_name']}_{cfg['model_type']}_run{cfg['run']}/model_full_{t}"
    best_val_loss = float("inf")
    resume_training = cfg['resume']
    last_ckpt_path = os.path.join(cfg.get('save_dir'), "last_checkpoint.pt")        
    global_lr_history = []
    if resume_training:
        resume_path = cfg["resume_checkpoint_path"]
        print(f"Resuming from checkpoint: {resume_path}")
        epoch, best_val_loss, training_metric, validation_metric, global_lr_history, counter = load_checkpoint(customBERT, optimizer, resume_path, device)
        print("best validtion loss", best_val_loss)
        
    start_epoch = epoch if resume_training else 0
    wait_time = cfg.get("wait_time", 12)
    counter = counter if resume_training else 0      
    for epoch in range(start_epoch, num_epochs):  
        return_metric = train_and_validate(customBERT, train_loader, val_loader, optimizer, criterion, scheduler=scheduler, debug=IS_DEBUG)
        # val_loss, val_acc, val_logits, val_preds, val_labels = validate(customBERT, train_loader, device, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {return_metric['train_loss']:.4f}, Train Acc: {return_metric['train_acc']:.4f}, Val Loss: {return_metric['val_loss']:.4f}, Val Acc: {return_metric['val_acc']:.4f}")
        training_metric["loss"].append(return_metric['train_loss'])
        training_metric["accuracy"].append(return_metric['train_acc'])
        validation_metric["loss"].append(return_metric['val_loss'])
        validation_metric["accuracy"].append(return_metric['val_acc'])
        global_lr_history.extend(return_metric['lr_history'])
        
        if return_metric['val_loss'] < best_val_loss:
            best_val_loss = return_metric['val_loss']
            counter = 0
            _save_path = os.path.join(cfg.get('save_dir'), f"best_model.pt")
            torch.save(customBERT.state_dict(), _save_path)
            print(f"Saved new best model to '{_save_path}' (Val Loss: {best_val_loss:.4f})")
            print (f"Running test models")
            # test_loss, test_acc = test(customBERT, test_loader, device, criterion, cfg=cfg, n_train=n_train)
            # print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        else:
            counter += 1
            print(f"No improvement in validation loss. Counter: {counter}")
        save_checkpoint(customBERT, optimizer, epoch, best_val_loss, last_ckpt_path, training_metric, validation_metric, global_lr_history, counter)
        if counter >= wait_time:
            print(f"Early stopping triggered: no improvement for {wait_time} consecutive epochs.")
            break
        
        df_dict = {}
        
        for k, v in training_metric.items():
            df_dict[f"train_{k}"] = v

        for k, v in validation_metric.items():
            df_dict[f"val_{k}"] = v
        
        if IS_DEBUG:
            print (f"EPOCH TRAINING METRIC: {training_metric}")
            print (f"EPOCH VALIDATION METRIC: {validation_metric}")

        pd.DataFrame(df_dict).to_csv(os.path.join(cfg.get('results'), f"epoch_training_metrics.csv"), index=False)
        
    
    plot_training_curves(training_metric, validation_metric,n_train=n_train, cfg=cfg)
    final_test_metric = test(customBERT, test_loader, epoch, criterion, cfg=cfg, n_train=n_train, best_model=False, debug=IS_DEBUG)
    print(f"Final Model -> Test Loss: {final_test_metric['loss']:.4f}, Test Acc: {final_test_metric['accuracy']:.4f}")

    print(f"Loading best model from {_save_path}")
    customBERT.load_state_dict(torch.load(os.path.join(_save_path)))
    customBERT.to(device)
    best_test_metric = test(customBERT, test_loader, epoch, criterion, cfg, n_train=n_train, best_model=True, debug=IS_DEBUG)
    print(f"Best Model -> Test Loss: {best_test_metric['loss']:.4f}, Test Acc: {best_test_metric['accuracy']:.4f}")
    with open(os.path.join(cfg.get('test_result_dir'), f"test_results.txt"), "w") as f:
        f.write(f"{epoch}\n")
        f.write("Final Model Results:\n")
        for k, v in final_test_metric.items():
            f.write(f"{k}: {v}\n")
        f.write("\nBest Model Results:\n")
        for k, v in best_test_metric.items():
            f.write(f"{k}: {v}\n")

    plt.figure()
    plt.plot(global_lr_history, 'go', linewidth=2)
    plt.xlabel("Epochs", fontsize =18)
    plt.ylabel("Learning Rate", fontsize=18)
    # plt.title("Learning rate profile", fontsize=20)
    plt.grid(True)
    lr_plot_path = os.path.join(cfg.get('save_dir'), "learning_rate_plot.pdf")
    plt.savefig(lr_plot_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved learning rate plot to {lr_plot_path}")
    

    end_time = time.perf_counter()
    print(f"total time taken: {end_time - init_time:.2f} seconds")
           
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate the model for jet classification.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the JSON config file.")
    # parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save log files")

    args = parser.parse_args()
    
    # log_dir = args.log_dir
    # os.makedirs(log_dir, exist_ok=True)
    
            
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # out_log_file = os.path.join(log_dir, f"output_{timestamp}.txt")
    # err_log_file = os.path.join(log_dir, f"error_{timestamp}.txt")
    # print(f"Output will be logged to {out_log_file}")
    # print(f"Errors will be logged to {err_log_file}")
    # # --- Redirect stdout and stderr ---
    # sys.stdout = open(out_log_file, "a")
    # sys.stderr = open(err_log_file, "a")

    cfg = OmegaConf.load(args.cfg)
    print(f"Configuration loaded from {args.cfg}")
    main(cfg)
    
