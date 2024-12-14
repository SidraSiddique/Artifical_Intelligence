def user_turn():
    position = int(input("Enter the position from [1...9]: "))
    while board[position - 1] != 0 or position > 9 or position < 1:
        position = int(input("Enter a valid move from [1...9]: "))
    board[position - 1] = -1 

def alpha_beta(player, alpha, beta):
    score = analyze_board()
    if score != 0:
        return score * player  
    if 0 not in board: 
        return 0

    if player == 1: 
        max_eval = -float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = player
                eval = alpha_beta(-player, alpha, beta)  
                board[i] = 0 
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  
        return max_eval
    else:  
        min_eval = float('inf')
        for i in range(9):
            if board[i] == 0:
                board[i] = player
                eval = alpha_beta(-player, alpha, beta) 
                board[i] = 0  
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  
        return min_eval

def computer_turn():
    best_score = -float('inf')
    best_move = -1
    
    for i in range(9):
        if board[i] == 0:  
           
            board[i] = -1 
            score = alpha_beta(1, -float('inf'), float('inf'))  
            board[i] = 0  
            if score > best_score:
                best_score = score
                best_move = i


    if best_move != -1: 
        board[best_move] = 1  

def analyze_board():
    winning_pos = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6],
        [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]
    ]
    for pos in winning_pos:
        if board[pos[0]] != 0 and board[pos[0]] == board[pos[1]] == board[pos[2]]:
            return board[pos[0]] 
    return 0 

def print_board():
    print("<-- CURRENT STATE -->")
    for i in range(0, 9, 3):
        print(" | ".join(['O' if board[i + j] == 1 else 'X' if board[i + j] == -1 else '-' for j in range(3)]))
    print()

if __name__ == '__main__':
    board = [0] * 9  
    print("Computer: O \t You: X")
    choice = int(input("Do you want to play 1(st) or 2(nd)? Enter 1 or 2: "))
    
    for turn in range(9):
        if analyze_board() != 0:
            break
        if (turn + choice) % 2 == 0:  
            computer_turn()
        else:  
            print_board()
            user_turn()
    
    print_board()
    winner = analyze_board()
    if winner == 0:
        print("\t--- DRAW ----")
    elif winner == -1:
        print("\tYou win, Computer loses!")
    else:
        print("\tComputer WINS")
