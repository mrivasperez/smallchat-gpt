'''
This file is part of SmallchatGPT.

SmallchatGPT is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

SmallchatGPT is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with SmallchatGPT. If not, see <https://www.gnu.org/licenses/>.
'''

import torch
import signal
import sys

# Define a flag to indicate if the signal has been received
interrupted = False

# Signal handler function


def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("\nSignal received. Saving model and exiting...")

    # Access the global model, optimizer, etc. from smallchat.py
    # Assuming smallchat.py is in the same directory or in a directory in your Python path
    from smallchat import model, optimizer, model_save_path, epoch, best_val_loss

    # Save the model and other necessary variables
    save_checkpoint(model, optimizer, epoch,
                    model_save_path, val_loss=best_val_loss)

    # Terminate the process
    sys.exit(0)  # Exit with a success code (0)


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


def save_checkpoint(model, optimizer, epoch, file_path, batch_idx=None, val_loss=None):
    """Saves a checkpoint of the model and optimizer state."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'batch_idx': batch_idx,
        'val_loss': val_loss
    }
    torch.save(checkpoint, file_path)
    print(f"Model checkpoint saved to {file_path}")
