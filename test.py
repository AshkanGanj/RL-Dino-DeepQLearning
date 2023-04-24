import argparse
import torch

from src.model import DeepQNetwork
from src.env import ChromeDino
import cv2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--fps", type=int, default=60,
                        help="frames per second")

    args = parser.parse_args()
    return args

def show_processed_img(state):
        # Convert the processed state to a NumPy array
        processed_state = state.squeeze(0).cpu().numpy()

        # Resize the image
        scale_factor = 2  # Modify this value to change the size of the image
        resized_state = cv2.resize(processed_state, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # rotate the image 90 degrees non clockwise
        rotated_state = cv2.rotate(resized_state, cv2.ROTATE_90_CLOCKWISE)

        # Display the rotated image in a window
        cv2.imshow("Processed State", rotated_state)

def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = DeepQNetwork()
    checkpoint_path = "{}/chrome_dino.pth".format(opt.saved_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    env = ChromeDino()
    state, raw_state, _, _ = env.step(0, True)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :]
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()
    done = False

    while not done:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()
        next_state, raw_next_state, reward, done = env.step(action, True)
        env.gamespeed += 0.001
        if torch.cuda.is_available():
            next_state = next_state.cuda()
        show_processed_img(next_state)
        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]  # concatenate the new frame with the previous 3 frames
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        state = next_state
    cv2.destroyAllWindows()
if __name__ == "__main__":
    opt = get_args()
    test(opt)
