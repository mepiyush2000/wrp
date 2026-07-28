import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

def train_flow_matching_unet(X_train, y_train, X_val, y_val, model, device, num_epochs=20, batch_size=64, lr=1e-3):
    print(f"Training Flow Matching U-Net on: {device}")
    model = model.to(device)

    # Initialize the EMA Model
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Pre-computing fixed validation noise and time...")
    fixed_val_data = []
    for batch_X, batch_y in val_loader:
        b_size = batch_X.size(0)
        fixed_x_0 = torch.randn_like(batch_y)
        # Stratified time sampling
        fixed_t = torch.linspace(0.01, 0.99, b_size) 
        fixed_t = fixed_t[torch.randperm(b_size)] 
        fixed_val_data.append((batch_X, batch_y, fixed_x_0, fixed_t))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Calculate total batches for the OneCycleLR
    total_steps = num_epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr, 
        total_steps=total_steps,
        pct_start=0.1,  # Spend 10% of training warming up
        anneal_strategy='cos'
    )
    
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss_accum = 0.0 # Changed name slightly for clarity

        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_X, x_1 = batch_X.to(device), batch_y.to(device)
            batch_size_current = batch_X.size(0)

            # --- CFG Context Dropout (UNCOMMENTED) ---
            # drop = (torch.rand(batch_size_current, device=device) < 0.10).view(-1, 1, 1, 1)
            # batch_X = torch.where(drop, torch.zeros_like(batch_X), batch_X) 

            # Generate Pure Noise & Time
            x_0 = torch.randn_like(x_1)
            t = torch.rand(batch_size_current, device=device)
            # normal_noise = torch.randn(batch_size_current, device=device)
            # t = torch.sigmoid(normal_noise)
            t_expanded = t.view(batch_size_current, 1, 1, 1)

            # Rectified Flow Interpolation
            x_t = (t_expanded * x_1) + ((1.0 - t_expanded) * x_0)
            target_velocity = x_1 - x_0

            # Forward & Optimize
            optimizer.zero_grad()
            predicted_velocity = model(batch_X, noisy_path=x_t, t=t)
            
            loss = F.mse_loss(predicted_velocity, target_velocity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step()

            # EMA Update
            ema_model.update_parameters(model)

            train_loss_accum += loss.item() * batch_size_current

        # ---------------------------
        # VALIDATION PHASE (Using EMA Model)
        # ---------------------------
        ema_model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for batch_X, batch_y, fixed_x_0, fixed_t in fixed_val_data:
                batch_X, x_1 = batch_X.to(device), batch_y.to(device)
                x_0 = fixed_x_0.to(device)
                t = fixed_t.to(device)
                
                batch_size_current = batch_X.size(0)
                t_expanded = t.view(batch_size_current, 1, 1, 1)

                x_t = (t_expanded * x_1) + ((1.0 - t_expanded) * x_0)
                target_velocity = x_1 - x_0

                # Validate using the EMA model
                predicted_velocity = ema_model(batch_X, noisy_path=x_t, t=t)
                loss = F.mse_loss(predicted_velocity, target_velocity)
                
                val_loss_accum += loss.item() * batch_size_current

        # --- LOGGING FIX: Calculate Epoch Averages and Append ---
        epoch_train_loss = train_loss_accum / len(train_loader.dataset)
        epoch_val_loss = val_loss_accum / len(val_loader.dataset)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"EMA Val Loss: {epoch_val_loss:.4f}")

        # Save the best EMA model
        if epoch_val_loss <= best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(ema_model.module.state_dict(), 'best_wrp_flow_model_ema.pth')
            print(f"Saved new best EMA model at epoch {epoch+1}")

        

    print("Training Complete.")
    return ema_model.module, train_losses, val_losses