import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TinyPCN
from parse_pgn import ChessDataset
from pathlib import Path

def train_from_pgn(
    data_path,
    model_save_path="chess_model.pth",
    batch_size=256,
    epochs=10,
    learning_rate=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train chess model from PGN-derived dataset.
    
    Args:
        data_path: Path to .pkl dataset file
        model_save_path: Where to save trained model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
    """
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = ChessDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    print(f"Dataset loaded: {len(dataset)} positions")
    
    # Initialize model
    model = TinyPCN(board_channels=18, policy_size=4672)
    model = model.to(device)
    
    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        for batch_idx, (boards, policy_targets, value_targets) in enumerate(dataloader):
            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            # Forward pass
            policy_logits, value_pred = model(boards)
            
            # Compute losses
            policy_loss = policy_criterion(policy_logits, policy_targets)
            value_loss = value_criterion(value_pred, value_targets)
            
            # Combined loss (you can adjust weights)
            loss = policy_loss + value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_policy = policy_loss_sum / (batch_idx + 1)
                avg_value = value_loss_sum / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {avg_loss:.4f} "
                      f"(Policy: {avg_policy:.4f}, Value: {avg_value:.4f})")
        
        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        avg_policy = policy_loss_sum / len(dataloader)
        avg_value = value_loss_sum / len(dataloader)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs} Complete")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Policy Loss: {avg_policy:.4f} | Value Loss: {avg_value:.4f}")
        print(f"{'='*60}\n")
        
        # Save checkpoint
        checkpoint_path = f"{model_save_path}.epoch{epoch+1}"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining complete! Model saved to {model_save_path}")


if __name__ == "__main__":
    # Train from PGN dataset
    train_from_pgn(
        data_path="chess_training_data.pkl",
        model_save_path="chess_model.pth",
        batch_size=256,
        epochs=10,
        learning_rate=0.001
    )
