'''
This file is part of SmallchatGPT.

SmallchatGPT is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

SmallchatGPT is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with SmallchatGPT. If not, see <https://www.gnu.org/licenses/>.
'''

import sys
import torch
import signal

# Define a flag to indicate if the signal has been received
interrupted = False


# Signal handler function (called by OS when SIGINT is received)
def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("\nSignal received. Saving model and exiting...")


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


def save_checkpoint(model, optimizer, epoch, file_path, batch_idx=None, val_loss=None):
    """Saves a checkpoint of the model and optimizer state."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'batch_idx': batch_idx,  # Optional
        'val_loss': val_loss    # Optional
    }
    torch.save(checkpoint, file_path)
    print(f"Model checkpoint saved to {file_path}")
    sys.exit(0)
