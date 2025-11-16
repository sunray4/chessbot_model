# Running Self-Play on Modal

## Setup

1. **Install Modal**:
   ```bash
   pip install modal
   ```

2. **Create Modal account** (if you don't have one):
   - Visit https://modal.com
   - Sign up for free account

3. **Authenticate Modal**:
   ```bash
   modal token new
   ```
   This will open a browser to authenticate.

## Running Self-Play

### One-time run:
```bash
modal run modal_selfplay.py
```

### Schedule continuous self-play:
```bash
modal deploy modal_selfplay.py
```

### Run multiple parallel games:
Modify the script to spawn multiple workers:
```python
@app.local_entrypoint()
def main():
    # Run 5 parallel workers, 10 games each
    futures = [run_selfplay_games.spawn(num_games=10) for _ in range(5)]
    results = [f.get() for f in futures]
    # Combine all results...
```

## Cost Estimates
- Modal free tier: 30 free credits/month
- Self-play uses minimal CPU, so very cheap
- Estimated cost: ~$0.10-0.50 per 100 games

## Next Steps
1. Collect self-play data
2. Download and use for training locally
3. Upload trained model back to Modal
4. Generate more games with improved model
