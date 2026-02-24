import torch
from src.data.celeba import CelebADataset

def test_dataset():
    print("Testing CelebADataset...")
    
    # 1. Initialize dataset
    # Make sure 'root' points to where your data actually is
    try:
        ds = CelebADataset(root="./data/celeba-subset", split="train", augment=False)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"✓ Dataset loaded. Length: {len(ds)}")
    
    # 2. Fetch one item
    item = ds[0]
    
    # 3. Check Tuple
    if isinstance(item, tuple):
        print("✓ Output is a tuple (Image, Label)")
    else:
        print(f"❌ Output is {type(item)} (Expected tuple). Did you update __getitem__?")
        return

    img, label = item
    
    # 4. Check Shapes
    print(f"  Image Shape: {img.shape} (Expected [3, 64, 64])")
    print(f"  Label Shape: {label.shape} (Expected [40])")
    
    # 5. Check Values
    print(f"  First 5 attribute values: {label[:5].tolist()}")
    
    # Verify binary nature
    is_binary = torch.all(torch.logical_or(label == 0, label == 1))
    if is_binary:
        print("✓ Labels are correctly formatted as binary (0.0 or 1.0)")
    else:
        print("❌ Labels contain values other than 0 and 1!")
        print("  (Did you forget the (df.values + 1) // 2 conversion?)")

if __name__ == "__main__":
    test_dataset()