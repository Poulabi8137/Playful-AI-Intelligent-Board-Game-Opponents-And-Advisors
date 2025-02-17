import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    openai.api_key = api_key
else:
    print("⚠️ OpenAI API key not found! Please check your .env file.")

# Tic-Tac-Toe Board Functions
def print_board(board):
    for row in board:
        print(" | ".join(row))
    print("-" * 9)

def check_winner(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True
    
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    
    return False

def available_moves(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == " "]

# Neural Network AI Model
class TicTacToeAI(nn.Module):
    def __init__(self):
        super(TicTacToeAI, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def board_to_tensor(board):
    return torch.tensor([1 if cell == 'O' else -1 if cell == 'X' else 0 for row in board for cell in row], dtype=torch.float32)

def best_move(board, model):
    moves = available_moves(board)
    if not moves:
        return None
    board_tensor = board_to_tensor(board).unsqueeze(0)
    with torch.no_grad():
        predictions = model(board_tensor).squeeze()
    
    move_scores = [(predictions[r * 3 + c].item(), (r, c)) for r, c in moves]
    best_index = max(move_scores, key=lambda x: x[0])[1]
    return best_index

def train_model(model, optimizer, criterion, epochs=1000):
    for epoch in range(epochs):
        board = [[' ' for _ in range(3)] for _ in range(3)]
        state = board_to_tensor(board).unsqueeze(0)
        target = torch.zeros(9)  # Changed from random values to zeros for stability
        optimizer.zero_grad()
        output = model(state).squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print("Training completed!")

# ✅ Improved GPT Advice Function with Error Handling & Backup
def get_gpt_advice(board):
    prompt = f"Given this Tic-Tac-Toe board state:\n{board}, what is the best move for 'X'?"
    max_retries = 3
    wait_time = 1  # Initial wait time for retry

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Try GPT-4 first
                messages=[{"role": "system", "content": "You are an expert Tic-Tac-Toe advisor."},
                          {"role": "user", "content": prompt}]
            )
            return response["choices"][0]["message"]["content"].strip()

        except openai.error.InvalidRequestError:
            print("❌ Model `gpt-4` does not exist or is unavailable. Switching to `gpt-3.5-turbo`...")
            return get_gpt_advice_backup(board)

        except openai.error.RateLimitError:
            print(f"⚠️ Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time *= 2  # Exponential backoff

        except openai.error.AuthenticationError:
            print("❌ Invalid API key! Please check your OpenAI account.")
            return "Invalid API key."

        except openai.error.OpenAIError as e:
            print(f"⚠️ OpenAI API error: {e}")
            return "Error in retrieving advice."

    print("❌ Maximum retries exceeded. No AI advice available.")
    return "No AI advice available."

# ✅ Backup: Use `gpt-3.5-turbo` if `gpt-4` is unavailable
def get_gpt_advice_backup(board):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an expert Tic-Tac-Toe advisor."},
                      {"role": "user", "content": f"Given this board: {board}, what is the best move for 'X'?"}]
        )
        return response["choices"][0]["message"]["content"].strip()

    except openai.error.RateLimitError:
        return "Rate limit exceeded. Upgrade your OpenAI plan."

    except Exception as e:
        print(f"⚠️ Error using GPT-3.5-turbo: {e}")
        return "No AI advice available."

# ✅ Local AI Model Fallback
def local_ai_advice(board):
    print("⚠️ Using local AI instead of OpenAI API.")
    model = TicTacToeAI()
    return f"Local AI suggests: {best_move(board, model)}"

# Game Loop
def play_game():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    model = TicTacToeAI()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    train_model(model, optimizer, criterion)
    
    print("🎮 Welcome to AI Tic-Tac-Toe!")
    print_board(board)

    while available_moves(board):
        try:
            user_input = input("Enter row and column (0-2): ")
            row, col = map(int, user_input.split())
        except ValueError:
            print("Invalid input! Please enter two numbers separated by a space.")
            continue
        
        if row not in range(3) or col not in range(3) or board[row][col] != ' ':
            print("Invalid move! Try again.")
            continue
        board[row][col] = 'X'

        if check_winner(board, 'X'):
            print_board(board)
            print("🎉 You win!")
            return
        
        if not available_moves(board):
            print("It's a tie!")
            return

        print("🤖 AI Advisor suggests:")
        ai_advice = get_gpt_advice(board) if api_key else local_ai_advice(board)
        print(ai_advice)

        ai_move = best_move(board, model)
        if ai_move:
            board[ai_move[0]][ai_move[1]] = 'O'
        print("🤖 AI moves:")
        print_board(board)

        if check_winner(board, 'O'):
            print("😢 AI wins! Better luck next time.")
            return

if __name__ == "__main__":
    play_game()
