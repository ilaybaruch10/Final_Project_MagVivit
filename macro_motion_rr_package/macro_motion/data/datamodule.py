from torch.utils.data import DataLoader
from .dataset import MacroMotionDataset

def make_loaders(metadata_csv, input_size, clip_frames, batch_size, num_workers, include_temp,
                 frame_stride=1):
    train_ds = MacroMotionDataset(metadata_csv, "train", input_size, clip_frames, True, include_temp, frame_stride)
    val_ds = MacroMotionDataset(metadata_csv, "val", input_size, clip_frames, True, include_temp, frame_stride)
    test_ds = MacroMotionDataset(metadata_csv, "test", input_size, clip_frames, True, include_temp, frame_stride)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )
